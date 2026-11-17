//! Tauri commands and the view-model DTOs they return.
//!
//! The DTOs here are deliberately decoupled from sequitur's own types: the
//! frontend gets a flat, JSON-friendly shape it can render directly and
//! addresses entities only by id. The `display`ed appearance is always the
//! PROJECTED state (canonical + staged intents); `pending` carries the
//! compact diff back to canonical.

use crate::state::{Canonical, Entity, EntityId, IntentOp, Projected, Registry};
use sequitur::{EntryKind, FileOperation, FileSequence, OperationPlan};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::ipc::Channel;
use tauri::State;

pub type AppState<'a> = State<'a, Mutex<Registry>>;

/// Serializes commits and freezes the registry while one runs. Because
/// `commit_pending` releases the registry lock during execution, the
/// registry-mutating commands check `in_flight` and refuse to run mid-commit
/// — that keeps the canonical snapshot the executor is acting on from drifting
/// out from under it, and makes the post-commit finalize sound.
#[derive(Default)]
pub struct CommitState {
    in_flight: AtomicBool,
    cancel: Arc<AtomicBool>,
}

/// Cancellation flag for the recursive scan (`search_open`), set by
/// `cancel_search` and polled by the walk so a scan of a huge/slow tree can be
/// abandoned promptly.
#[derive(Default)]
pub struct SearchState {
    cancel: Arc<AtomicBool>,
}

/// Live progress from a running scan: the directory currently being walked and
/// running tallies, so the UI can show it's still working.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchProgress {
    pub dir: String,
    pub scanned: usize,
    pub matches: usize,
}

/// A running H.264 export: the ffmpeg child (so `cancel_export` can kill it
/// mid-encode) and whether the kill was user-requested.
#[derive(Default)]
pub struct ExportState {
    child: Arc<Mutex<Option<std::process::Child>>>,
    cancelled: Arc<AtomicBool>,
}

const COMMIT_BUSY: &str = "a commit is in progress; try again when it finishes";

/// A parsed sequence collapsed into a single visual entity.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SequenceDto {
    /// Hash-notation display name, e.g. `shot_###.exr`.
    pub name: String,
    pub prefix: String,
    /// Separator between prefix and frame number (e.g. `_` or `.`), if any.
    pub delimiter: Option<String>,
    /// Text between the frame number and the extension (e.g. `_left`), if any.
    pub suffix: Option<String>,
    pub extension: String,
    /// First and last frame of the nominal span.
    pub first_frame: i32,
    pub last_frame: i32,
    /// Length of the nominal span (`last - first + 1`).
    pub frame_count: i32,
    /// How many frames are actually present on disk.
    pub actual_frame_count: usize,
    /// Total number of frames missing from the span (drives the "incomplete"
    /// badge). Computed without materialising them.
    pub missing_frame_count: u64,
    /// The missing frame numbers, capped at `MISSING_FRAMES_CAP`. Complete for
    /// any real shot; a "sequence" of date-stamped photos can have a span of
    /// millions and must not be materialised (it froze the app).
    pub missing_frames: Vec<i32>,
    pub padding: usize,
}

/// Upper bound on missing frames listed per sequence in a DTO.
const MISSING_FRAMES_CAP: usize = 5000;

/// A loose filesystem entry that is not part of a sequence.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryDto {
    pub path: String,
    pub name: String,
}

/// Compact description of how an entity's staged state differs from disk.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingDto {
    /// e.g. `render_###.exr`
    pub before_summary: String,
    /// e.g. `comp_###.exr`, or `(deleted)`
    pub after_summary: String,
    /// One human-readable line per staged intent, in stage order; the index
    /// in this list is the index to pass to `revert_intent`.
    pub intents: Vec<String>,
}

/// One registry entity as the frontend sees it: projected appearance plus
/// lifecycle and the pending diff.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityView {
    pub id: EntityId,
    /// One of `sequence`, `file`, `directory`, `symlink`, `other`.
    pub kind: String,
    /// One of `unchanged`, `modified`, `deleted`, `departing`.
    pub lifecycle: String,
    pub hidden: bool,
    /// Projected appearance when this is a sequence.
    pub sequence: Option<SequenceDto>,
    /// Projected appearance when this is a loose entry.
    pub entry: Option<EntryDto>,
    pub pending: Option<PendingDto>,
    /// Paths this entity would contest with another entity's projection;
    /// non-empty means committing as staged would collide.
    pub conflicts: Vec<String>,
    /// Set when the staged intents no longer fold cleanly over the canonical
    /// snapshot (e.g. an external change shrank the frame set so a staged
    /// offset now underflows). The appearance shown is the *canonical* state;
    /// `brokenReason` explains why, and the entity is excluded from commits
    /// until the offending intent is reverted.
    pub broken: bool,
    pub broken_reason: Option<String>,
    /// Directory the entity canonically lives in (useful when a view mixes
    /// entities from several directories, e.g. search results).
    pub dir: String,
}

/// A directory's contents as registry entities, sequences first.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirView {
    pub path: String,
    pub entities: Vec<EntityView>,
    /// Names of entities whose staged edits were discarded because the
    /// underlying files changed on disk since they were staged.
    pub voided: Vec<String>,
}

impl SequenceDto {
    fn from_sequence(seq: &FileSequence) -> Self {
        Self {
            name: seq
                .sequence_string()
                .unwrap_or_else(|_| String::from("<sequence>")),
            prefix: seq.prefix().unwrap_or_default().to_string(),
            delimiter: seq.delimiter().ok().flatten().map(String::from),
            suffix: seq.suffix().ok().flatten().map(String::from),
            extension: seq.extension().unwrap_or_default().to_string(),
            first_frame: seq.first_frame(),
            last_frame: seq.last_frame(),
            frame_count: seq.frame_count(),
            actual_frame_count: seq.actual_frame_count(),
            missing_frame_count: seq.missing_frame_count(),
            missing_frames: seq.missing_frames_capped(MISSING_FRAMES_CAP),
            padding: seq.padding().unwrap_or(0),
        }
    }
}

fn kind_str(kind: EntryKind) -> &'static str {
    match kind {
        EntryKind::File => "file",
        EntryKind::Directory => "directory",
        EntryKind::Symlink => "symlink",
        EntryKind::Other => "other",
    }
}

/// Builds the projected appearance of a canonical snapshot (used to render a
/// broken entity, which has no usable projection).
fn canonical_appearance(
    canonical: &Canonical,
) -> (&'static str, bool, Option<SequenceDto>, Option<EntryDto>) {
    match canonical {
        Canonical::Sequence(seq) => ("sequence", false, Some(SequenceDto::from_sequence(seq)), None),
        Canonical::Loose { path, kind, hidden } => {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let dto = EntryDto {
                path: path.to_string_lossy().into_owned(),
                name,
            };
            (kind_str(*kind), *hidden, None, Some(dto))
        }
    }
}

/// Infallible: an entity whose intents no longer fold is rendered in its
/// canonical appearance with `broken` set, rather than failing the whole
/// listing.
fn entity_view(id: EntityId, entity: &Entity, conflicts: Vec<String>) -> EntityView {
    let dir = entity
        .canonical_dir()
        .map(|d| d.to_string_lossy().into_owned())
        .unwrap_or_default();
    let projection = match entity.project() {
        Ok(p) => p,
        Err(e) => {
            let (kind, hidden, sequence, entry) = canonical_appearance(&entity.canonical);
            return EntityView {
                id,
                kind: kind.to_string(),
                lifecycle: "modified".to_string(),
                hidden,
                sequence,
                entry,
                pending: Some(PendingDto {
                    before_summary: entity.canonical_name(),
                    after_summary: "(cannot apply)".to_string(),
                    intents: entity.intents.iter().map(IntentOp::summary).collect(),
                }),
                conflicts,
                broken: true,
                broken_reason: Some(e.to_string()),
                dir,
            };
        }
    };
    let lifecycle = entity.lifecycle(&projection);

    let (kind, hidden, sequence, entry, after_summary) =
        match (&entity.canonical, &projection.projected) {
            (Canonical::Sequence(_), Projected::Sequence(projected)) => {
                let dto = SequenceDto::from_sequence(projected);
                let after = dto.name.clone();
                ("sequence", false, Some(dto), None, after)
            }
            (Canonical::Loose { kind, hidden, .. }, Projected::Loose(path)) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let dto = EntryDto {
                    path: path.to_string_lossy().into_owned(),
                    name: name.clone(),
                };
                (kind_str(*kind), *hidden, None, Some(dto), name)
            }
            _ => unreachable!("projection kind always matches canonical kind"),
        };

    let pending = if entity.intents.is_empty() {
        None
    } else {
        let before_summary = if entity.is_copy() {
            format!("copy of {}", entity.canonical_name())
        } else {
            entity.canonical_name()
        };
        Some(PendingDto {
            before_summary,
            after_summary: if projection.deleted {
                "(deleted)".to_string()
            } else {
                after_summary
            },
            intents: entity.intents.iter().map(IntentOp::summary).collect(),
        })
    };

    EntityView {
        id,
        kind: kind.to_string(),
        lifecycle: lifecycle.as_str().to_string(),
        hidden,
        sequence,
        entry,
        pending,
        conflicts,
        broken: false,
        broken_reason: None,
        dir,
    }
}

/// Stage-time conflict set for one entity, as display strings.
fn conflicts_of(
    all: &std::collections::HashMap<EntityId, Vec<PathBuf>>,
    id: EntityId,
) -> Vec<String> {
    all.get(&id)
        .map(|paths| {
            paths
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn view_of(registry: &Registry, id: EntityId) -> Result<EntityView, String> {
    let entity = registry
        .get(id)
        .ok_or_else(|| format!("no entity with id {id}"))?;
    let conflicts = registry.projected_conflicts();
    Ok(entity_view(id, entity, conflicts_of(&conflicts, id)))
}

/// A virtual multi-directory view of arbitrary entities (the bin). Each
/// entity's own directory is re-scanned first so the listing reflects disk
/// (and stale intents get voided the same way a normal open would). Entities
/// that no longer exist are silently dropped. Because it comes back as a
/// `DirView`, every verb / selection / drag / thumbnail path works on it
/// exactly as on search results.
#[tauri::command]
pub fn view_entities(
    ids: Vec<EntityId>,
    state: AppState,
    commit: State<CommitState>,
) -> Result<DirView, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let mut registry = state.lock().map_err(|e| e.to_string())?;

    // Refresh each distinct directory once, so canonical state is current.
    let mut dirs: Vec<PathBuf> = ids
        .iter()
        .filter_map(|id| registry.get(*id).and_then(|e| e.canonical_dir()))
        .collect();
    dirs.sort();
    dirs.dedup();
    let mut voided = Vec::new();
    for dir in dirs {
        if let Ok(listing) = FileSequence::parse_directory(&dir, 2) {
            let entries = listing
                .entries
                .iter()
                .map(|e| (e.path.clone(), e.kind, e.hidden))
                .collect();
            let outcome = registry.absorb_scan(&dir, listing.sequences, entries);
            voided.extend(outcome.voided);
        }
    }

    let conflicts = registry.projected_conflicts();
    let entities = ids
        .into_iter()
        .filter_map(|id| {
            registry
                .get(id)
                .map(|entity| entity_view(id, entity, conflicts_of(&conflicts, id)))
        })
        .collect();
    Ok(DirView {
        path: String::from("bin"),
        entities,
        voided,
    })
}

/// Scans `path` into the registry (reconciling identity against previous
/// scans) and returns its entities, sequences first.
///
/// `min_frames` defaults to 2: a group of fewer than that many frames is
/// surfaced as loose files rather than a sequence.
#[tauri::command]
pub fn open_directory(
    path: String,
    min_frames: Option<usize>,
    state: AppState,
    commit: State<CommitState>,
) -> Result<DirView, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let dir = dunce::canonicalize(PathBuf::from(&path)).map_err(|e| e.to_string())?;
    let listing = FileSequence::parse_directory(&dir, min_frames.unwrap_or(2))
        .map_err(|e| e.to_string())?;
    let entries = listing
        .entries
        .iter()
        .map(|e| (e.path.clone(), e.kind, e.hidden))
        .collect();

    let mut registry = state.lock().map_err(|e| e.to_string())?;
    let outcome = registry.absorb_scan(&dir, listing.sequences, entries);
    let mut ids = outcome.ids;
    // Staged copies landing in this directory are shown where they'll appear.
    ids.extend(registry.copies_in(&dir));

    let conflicts = registry.projected_conflicts();
    let entities = ids
        .into_iter()
        .filter_map(|id| {
            registry
                .get(id)
                .map(|entity| entity_view(id, entity, conflicts_of(&conflicts, id)))
        })
        .collect();

    Ok(DirView {
        path: dir.to_string_lossy().into_owned(),
        entities,
        voided: outcome.voided,
    })
}

/// Stages a generative copy of `id` (new name via `prefix` for sequences or
/// `name` for loose files, and/or a new home via `dest`). Returns the view of
/// the newly minted copy entity.
#[tauri::command]
pub fn stage_copy(
    id: EntityId,
    dest: Option<String>,
    prefix: Option<String>,
    name: Option<String>,
    state: AppState,
    commit: State<CommitState>,
) -> Result<EntityView, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let mut registry = state.lock().map_err(|e| e.to_string())?;
    let copy_id = registry.stage_copy(id, dest.map(PathBuf::from), prefix, name)?;
    view_of(&registry, copy_id)
}

/// Stages an intent on an entity. Validated eagerly: an impossible edit is
/// rejected now, not at commit. Returns the entity's refreshed view.
#[tauri::command]
pub fn stage_intent(
    id: EntityId,
    intent: IntentOp,
    state: AppState,
    commit: State<CommitState>,
) -> Result<EntityView, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let mut registry = state.lock().map_err(|e| e.to_string())?;
    registry.stage_intent(id, intent)?;
    view_of(&registry, id)
}

/// Removes the staged intent at `index` on `id` (per-item revert). Returns
/// `None` when the revert cancelled an uncommitted copy (the entity is gone).
#[tauri::command]
pub fn revert_intent(
    id: EntityId,
    index: usize,
    state: AppState,
    commit: State<CommitState>,
) -> Result<Option<EntityView>, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let mut registry = state.lock().map_err(|e| e.to_string())?;
    let removed = registry.revert_intent(id, index)?;
    if removed {
        return Ok(None);
    }
    view_of(&registry, id).map(Some)
}

/// Drops every staged intent on `id`. Returns `None` when this discarded an
/// uncommitted copy (the entity is gone).
#[tauri::command]
pub fn clear_intents(
    id: EntityId,
    state: AppState,
    commit: State<CommitState>,
) -> Result<Option<EntityView>, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let mut registry = state.lock().map_err(|e| e.to_string())?;
    let removed = registry.clear_intents(id)?;
    if removed {
        return Ok(None);
    }
    view_of(&registry, id).map(Some)
}

/// Every entity with staged intents, across all directories seen this
/// session (the review tray).
#[tauri::command]
pub fn pending_entities(state: AppState) -> Result<Vec<EntityView>, String> {
    let registry = state.lock().map_err(|e| e.to_string())?;
    let conflicts = registry.projected_conflicts();
    Ok(registry
        .pending()
        .into_iter()
        .map(|(id, entity)| entity_view(id, entity, conflicts_of(&conflicts, id)))
        .collect())
}

/// One executor step, streamed to the frontend progress bar. `index`/`total`
/// count scheduled steps, which include synthetic temp-hop renames.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressDto {
    pub index: usize,
    pub total: usize,
    pub label: String,
}

/// The outcome of a commit attempt that was allowed to start (pre-flight
/// conflicts are returned as a command error instead, with nothing run).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitOutcome {
    /// True only if every operation ran and the registry was finalized.
    pub committed: bool,
    /// Operations that completed (including any later rolled-back temp hops).
    pub executed: usize,
    pub cancelled: bool,
    /// Entities skipped because their intents no longer fold (broken).
    pub skipped: usize,
    /// The first failure, if one stopped the run.
    pub error: Option<String>,
}

/// Commits staged intents to disk: builds the minimal-diff plan across the
/// chosen entities (all pending if `ids` is `None`) and runs it through
/// sequitur's cycle-safe executor, streaming per-step progress.
///
/// Serialized: only one commit runs at a time, and while it runs the registry
/// is frozen (the mutating commands reject with [`COMMIT_BUSY`]) so the
/// canonical snapshot the executor acts on cannot drift.
///
/// Draft-mode contract: pre-flight conflicts abort with an error before
/// anything runs; broken entities are skipped. On full success the targeted
/// entities are finalized (canonical = projected, intents drained). On
/// failure or cancellation the targeted entities' intent logs are **cleared**
/// — some of their operations may have already hit disk, so replaying the
/// intents would double-apply; the frontend re-scans to re-establish truth.
#[tauri::command]
pub async fn commit_pending(
    ids: Option<Vec<EntityId>>,
    force: Option<bool>,
    progress: Channel<ProgressDto>,
    state: State<'_, Mutex<Registry>>,
    commit: State<'_, CommitState>,
) -> Result<CommitOutcome, String> {
    // Acquire the single-commit guard; bail if one is already running.
    if commit.in_flight.swap(true, Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let outcome = run_commit(ids, force, progress, &state, &commit).await;
    commit.in_flight.store(false, Ordering::SeqCst);
    outcome
}

async fn run_commit(
    ids: Option<Vec<EntityId>>,
    force: Option<bool>,
    progress: Channel<ProgressDto>,
    state: &Mutex<Registry>,
    commit: &CommitState,
) -> Result<CommitOutcome, String> {
    // Snapshot the work under the lock, then execute without holding it.
    // A broken entity (commit_plan errors) is skipped, not fatal.
    let (plan, target_ids, skipped) = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        let mut plan = OperationPlan::new();
        let mut target_ids = Vec::new();
        let mut skipped = 0usize;
        for (id, entity) in registry.pending() {
            if ids.as_ref().is_some_and(|ids| !ids.contains(&id)) {
                continue;
            }
            match entity.commit_plan() {
                Ok(p) => {
                    plan.extend(p);
                    target_ids.push(id);
                }
                Err(_) => skipped += 1,
            }
        }
        (plan, target_ids, skipped)
    };

    if target_ids.is_empty() {
        return Err(if skipped > 0 {
            "staged changes can't be applied (broken by an external change); revert them"
                .to_string()
        } else {
            "nothing staged to commit".to_string()
        });
    }

    let flag = commit.cancel.clone();
    flag.store(false, Ordering::SeqCst);

    let result = tauri::async_runtime::spawn_blocking(move || {
        plan.commit_observed(force.unwrap_or(false), |p| {
            let label = p
                .operation
                .destination()
                .unwrap_or_else(|| p.operation.source())
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let _ = progress.send(ProgressDto {
                index: p.index,
                total: p.total,
                label,
            });
            if flag.load(Ordering::SeqCst) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    let cancelled = commit.cancel.load(Ordering::SeqCst);
    let success = result.success() && !cancelled;
    {
        let mut registry = state.lock().map_err(|e| e.to_string())?;
        if success {
            registry.finalize_commit(&target_ids);
        } else {
            // Operations may have partially applied; discard the staged
            // intents of every targeted entity so a re-scan reconciles them
            // to disk reality instead of double-applying.
            for id in &target_ids {
                let _ = registry.clear_intents(*id);
            }
        }
    }

    Ok(CommitOutcome {
        committed: success,
        executed: result.executed.len(),
        cancelled,
        skipped,
        error: result
            .failed
            .first()
            .map(|(op, e)| format!("{}: {e}", op.source().display())),
    })
}

/// Signals the in-flight commit to stop before its next step.
#[tauri::command]
pub fn cancel_commit(commit: State<'_, CommitState>) {
    commit.cancel.store(true, Ordering::SeqCst);
}

/// Signals the in-flight recursive scan to stop; it returns whatever it has
/// found so far.
#[tauri::command]
pub fn cancel_search(search: State<'_, SearchState>) {
    search.cancel.store(true, Ordering::SeqCst);
}

/// Kills the running H.264 export; the partial output file is removed.
#[tauri::command]
pub fn cancel_export(export: State<'_, ExportState>) {
    export.cancelled.store(true, Ordering::SeqCst);
    if let Ok(mut slot) = export.child.lock() {
        if let Some(child) = slot.as_mut() {
            let _ = child.kill();
        }
    }
}

/// Encodes a sequence to an H.264 `.mp4` at `dest` with the vendored ffmpeg.
///
/// Frames are fed through a concat list, so missing frames simply play
/// through (no renumbering); EXR input gets an sRGB transfer on decode (not
/// the ACES view transform); odd dimensions are rounded up to even for
/// yuv420p. Progress streams as encoded-frame counts. Returns `true` when the
/// file was written, `false` when cancelled (partial output removed).
#[tauri::command]
pub async fn export_h264(
    id: EntityId,
    fps: f64,
    crf: u32,
    dest: String,
    overwrite: bool,
    progress: Channel<ProgressDto>,
    state: AppState<'_>,
    commit: State<'_, CommitState>,
    export: State<'_, ExportState>,
) -> Result<bool, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    if export.child.lock().map(|s| s.is_some()).unwrap_or(false) {
        return Err("an export is already running".to_string());
    }
    let paths: Vec<PathBuf> = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        let entity = registry
            .get(id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        match &entity.canonical {
            Canonical::Sequence(seq) => seq.items().iter().map(|i| i.path()).collect(),
            _ => return Err("export needs a sequence".to_string()),
        }
    };
    if paths.is_empty() {
        return Err("sequence has no frames".to_string());
    }
    let dest_path = PathBuf::from(&dest);
    if dest_path.exists() && !overwrite {
        return Err(format!("{dest} already exists"));
    }
    let is_exr = paths[0]
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("exr"))
        .unwrap_or(false);
    let ffmpeg = crate::thumbs::ffmpeg_path().ok_or("ffmpeg is unavailable")?;
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 24.0 };
    let crf = crf.clamp(0, 51);
    let total = paths.len();
    let label = dest_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "export".into());

    export.cancelled.store(false, Ordering::SeqCst);
    let child_slot = export.child.clone();
    let cancelled = export.cancelled.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<bool, String> {
        // Concat list: one line per existing frame with an explicit duration
        // (the final entry repeated per the demuxer's convention). Forward
        // slashes work on every platform; single quotes are escaped.
        let dur = 1.0 / fps;
        let quote = |p: &PathBuf| {
            p.to_string_lossy()
                .replace('\\', "/")
                .replace('\'', "'\\''")
        };
        let mut list = String::new();
        for p in &paths {
            list.push_str(&format!("file '{}'\nduration {dur:.6}\n", quote(p)));
        }
        if let Some(last) = paths.last() {
            list.push_str(&format!("file '{}'\n", quote(last)));
        }
        let list_path = std::env::temp_dir().join(format!(
            "seqmanip_export_{}_{id}.txt",
            std::process::id()
        ));
        fs::write(&list_path, list).map_err(|e| e.to_string())?;

        let mut cmd = crate::thumbs::hidden_command(&ffmpeg);
        cmd.args(["-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]);
        if is_exr {
            // Scene-linear -> sRGB on decode; without this the mp4 is "washed
            // out linear". The full ACES view lives in the app's own decoder,
            // not ffmpeg, so this is the honest approximation.
            cmd.args(["-apply_trc", "iec61966_2_1"]);
        }
        cmd.args(["-f", "concat", "-safe", "0", "-i"])
            .arg(&list_path)
            .args([
                "-vf",
                "scale=ceil(iw/2)*2:ceil(ih/2)*2",
                "-r",
                &format!("{fps}"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                &crf.to_string(),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ])
            .arg(&dest_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| format!("ffmpeg spawn: {e}"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        *child_slot.lock().unwrap() = Some(child);

        // Drain stderr off-thread so a chatty encoder can't deadlock us.
        let err_thread = std::thread::spawn(move || {
            use std::io::Read;
            let mut s = String::new();
            if let Some(mut e) = stderr {
                let _ = e.read_to_string(&mut s);
            }
            s
        });

        if let Some(out) = stdout {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if let Some(v) = line.strip_prefix("frame=") {
                    if let Ok(n) = v.trim().parse::<usize>() {
                        let _ = progress.send(ProgressDto {
                            index: n.min(total),
                            total,
                            label: label.clone(),
                        });
                    }
                }
            }
        }

        let status = {
            let taken = child_slot.lock().unwrap().take();
            match taken {
                Some(mut c) => c.wait().map_err(|e| e.to_string())?,
                None => return Ok(false), // cancel raced us and reaped it
            }
        };
        let errs = err_thread.join().unwrap_or_default();
        let _ = fs::remove_file(&list_path);

        if cancelled.load(Ordering::SeqCst) {
            let _ = fs::remove_file(&dest_path);
            return Ok(false);
        }
        if !status.success() {
            let _ = fs::remove_file(&dest_path);
            let msg = errs.trim();
            return Err(if msg.is_empty() {
                "ffmpeg failed".to_string()
            } else {
                msg.to_string()
            });
        }
        Ok(true)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Returns a small preview image for an entity as a `data:` URL, or `None`
/// when the format isn't displayable. Sequences preview their middle frame.
/// Decodes off the IPC thread; results are cached on disk by path + mtime.
#[tauri::command]
pub async fn thumbnail(
    id: EntityId,
    input: Option<crate::thumbs::InputSpace>,
    view: Option<crate::thumbs::ViewTransform>,
    app: tauri::AppHandle,
    state: State<'_, Mutex<Registry>>,
) -> Result<Option<String>, String> {
    use tauri::Manager;
    let color = color_opts(input, view, None);

    // Resolve the path under the lock, decode without it.
    let path = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        let entity = registry
            .get(id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        match &entity.canonical {
            Canonical::Sequence(seq) => {
                let items: Vec<PathBuf> = seq.items().iter().map(|i| i.path()).collect();
                crate::thumbs::representative_path(&items)
            }
            Canonical::Loose {
                path,
                kind: EntryKind::File,
                ..
            } => Some(path.clone()),
            _ => None,
        }
    };
    let Some(path) = path else { return Ok(None) };
    if !crate::thumbs::displayable(&path) {
        return Ok(None);
    }

    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("thumbs");

    tauri::async_runtime::spawn_blocking(move || {
        crate::thumbs::thumbnail_data_url(&path, &cache_dir, &color)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Builds [`ColorOpts`] from optional per-call overrides.
fn color_opts(
    input: Option<crate::thumbs::InputSpace>,
    view: Option<crate::thumbs::ViewTransform>,
    exposure: Option<f32>,
) -> crate::thumbs::ColorOpts {
    crate::thumbs::ColorOpts {
        input: input.unwrap_or_default(),
        view: view.unwrap_or_default(),
        exposure: exposure.unwrap_or(0.0).clamp(-8.0, 8.0),
    }
}

/// Evenly-spaced preview frames across a sequence, for hover-scrubbing its
/// tile. Returns up to `count` (default 12, clamped 2–24) data URLs in frame
/// order; non-sequences and undisplayable formats return an empty list.
/// Each frame goes through the same on-disk cache as [`thumbnail`], so a
/// sequence only ever decodes once.
#[tauri::command]
pub async fn scrub_thumbnails(
    id: EntityId,
    count: Option<usize>,
    input: Option<crate::thumbs::InputSpace>,
    view: Option<crate::thumbs::ViewTransform>,
    app: tauri::AppHandle,
    state: State<'_, Mutex<Registry>>,
) -> Result<Vec<String>, String> {
    use tauri::Manager;
    let color = color_opts(input, view, None);

    // A sequence scrubs across its frames; a loose movie file scrubs across
    // evenly-spaced timestamps.
    enum Source {
        Frames(Vec<PathBuf>),
        Movie(PathBuf),
    }
    let source = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        let entity = registry
            .get(id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        match &entity.canonical {
            Canonical::Sequence(seq) => {
                let items: Vec<PathBuf> = seq.items().iter().map(|i| i.path()).collect();
                if items.len() < 2 {
                    Source::Frames(Vec::new())
                } else {
                    let n = count.unwrap_or(12).clamp(2, 24).min(items.len());
                    Source::Frames(
                        (0..n)
                            .map(|i| items[i * (items.len() - 1) / (n - 1)].clone())
                            .collect(),
                    )
                }
            }
            Canonical::Loose {
                path,
                kind: EntryKind::File,
                ..
            } if crate::thumbs::is_movie(path) => Source::Movie(path.clone()),
            _ => Source::Frames(Vec::new()),
        }
    };

    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("thumbs");
    let samples = count.unwrap_or(12);

    match source {
        Source::Frames(paths) => {
            if paths.is_empty() || !crate::thumbs::displayable(&paths[0]) {
                return Ok(Vec::new());
            }
            tauri::async_runtime::spawn_blocking(move || {
                paths
                    .iter()
                    .map(|p| {
                        crate::thumbs::thumbnail_data_url(p, &cache_dir, &color)
                            .ok()
                            .flatten()
                            .unwrap_or_default()
                    })
                    .collect::<Vec<String>>()
            })
            .await
            .map_err(|e| e.to_string())
        }
        Source::Movie(path) => tauri::async_runtime::spawn_blocking(move || {
            crate::thumbs::movie_scrub_data_urls(&path, &cache_dir, samples)
        })
        .await
        .map_err(|e| e.to_string()),
    }
}

/// Recursively searches under `root` for sequences and movie files whose
/// name contains `query` (case-insensitive) and returns them as **real
/// registry entities** in a `DirView`, so search results behave exactly like
/// a directory listing: selectable, stageable, thumbnailed.
///
/// Every directory containing a match is absorbed into the registry (full
/// directory, so identity reconciliation stays consistent). Hidden dirs are
/// skipped and symlinked dirs are not followed (cycle-safe); depth, breadth
/// and result caps keep runaway trees bounded — `DirView.voided` is empty and
/// truncation is signalled by hitting the result cap exactly.
#[tauri::command]
pub async fn search_open(
    root: String,
    query: String,
    progress: Channel<SearchProgress>,
    state: State<'_, Mutex<Registry>>,
    commit: State<'_, CommitState>,
    search: State<'_, SearchState>,
) -> Result<DirView, String> {
    const MAX_RESULTS: usize = 200;
    const MAX_DIRS: usize = 2000;
    const MAX_DEPTH: usize = 12;

    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    // An empty query matches everything: "find all sequences and movies".
    let needle = query.trim().to_lowercase();
    let root = dunce::canonicalize(PathBuf::from(&root)).map_err(|e| e.to_string())?;

    // Fresh cancel flag for this scan.
    search.cancel.store(false, Ordering::SeqCst);
    let cancel = search.cancel.clone();
    // Always confirm in the log which code is running (a stale dev binary
    // silently lacking recent fixes cost real debugging time).
    eprintln!("[scan] start root={} query={:?}", root.display(), needle);

    // Results are collected into a shared vec so that if the walk is
    // abandoned (cancel, or a hard timeout because workers wedged in a
    // filesystem call) we still return whatever it found so far.
    let found: Arc<std::sync::Mutex<Vec<(PathBuf, sequitur::DirectoryListing)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let found_w = found.clone();

    // Progress is NOT sent from the worker threads. Eight threads hammering
    // the IPC channel from a hot loop can back-pressure the channel and stall
    // the whole runtime (workers block in send(); the async timeout starves;
    // the UI freezes at a fixed count — exactly the observed symptom). Instead
    // workers just record "where am I" into a shared slot and a single async
    // reporter emits it at a gentle rate.
    let cur_dir: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    let scanned_atomic = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let matches_shared = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cur_dir_w = cur_dir.clone();
    let scanned_w = scanned_atomic.clone();
    let matches_w = matches_shared.clone();

    // Walk without holding the registry lock; collect per-dir listings that
    // contain at least one match.
    let walk_root = root.clone();
    let walk_needle = needle.clone();
    let walk = tauri::async_runtime::spawn_blocking(move || {
            use std::sync::atomic::AtomicUsize;
            use std::time::{Duration, Instant};

            // Directories are parsed in parallel to hide per-directory latency:
            // on a network-backed filesystem (Dropbox / macOS File Provider)
            // each listing is a round-trip, and walking them one-at-a-time is
            // what made deep scans appear to hang. A wall-clock budget bounds
            // the very worst case — a scan returns partial results rather than
            // hanging forever.
            const WALK_THREADS: usize = 8;
            let deadline = Instant::now() + Duration::from_secs(20);

            // Directories that never hold image sequences but routinely hold
            // tens of thousands of files (build outputs, VCS, package caches).
            // Descending into them is pure cost — and on Dropbox each stat can
            // trigger an on-demand download, which is what made a scan appear
            // to hang inside a single folder.
            const SKIP_DIRS: &[&str] = &[
                "target",
                "node_modules",
                ".git",
                ".svn",
                ".hg",
                "__pycache__",
                ".venv",
                "venv",
                ".cache",
                "Library",
                "Pods",
                "DerivedData",
                "build",
                "dist",
                ".build",
                ".gradle",
                ".idea",
                ".vscode",
            ];
            let skip = |p: &std::path::Path| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| SKIP_DIRS.contains(&n))
                    .unwrap_or(false)
            };

            let mut frontier: Vec<(PathBuf, usize)> = vec![(walk_root, 0)];
            let matches_atomic = matches_w; // shared with the async reporter
            let mut dirs_seen = 0usize;

            while !frontier.is_empty() {
                let matches = matches_atomic.load(Ordering::Relaxed);
                if dirs_seen >= MAX_DIRS
                    || matches >= MAX_RESULTS
                    || Instant::now() >= deadline
                    || cancel.load(Ordering::Relaxed)
                {
                    break;
                }
                // Take at most the remaining dir budget from this level.
                let take = (MAX_DIRS - dirs_seen).min(frontier.len());
                let batch: Vec<(PathBuf, usize)> = frontier.drain(..take).collect();

                // Parse this batch of directories across a small thread pool.
                // Each worker reports the directory it's entering so the UI can
                // show progress, and bails promptly on cancel / deadline.
                let idx = AtomicUsize::new(0);
                let parsed =
                    std::sync::Mutex::new(Vec::<(PathBuf, usize, sequitur::DirectoryListing)>::new());
                std::thread::scope(|s| {
                    for _ in 0..WALK_THREADS {
                        s.spawn(|| loop {
                            let i = idx.fetch_add(1, Ordering::Relaxed);
                            if i >= batch.len()
                                || Instant::now() >= deadline
                                || cancel.load(Ordering::Relaxed)
                            {
                                break;
                            }
                            let (dir, depth) = &batch[i];
                            // Cheap: record position for the async reporter.
                            if let Ok(mut s) = cur_dir_w.lock() {
                                *s = dir.to_string_lossy().into_owned();
                            }
                            scanned_w.store(dirs_seen + i + 1, Ordering::Relaxed);
                            let t0 = Instant::now();
                            let res = FileSequence::parse_directory(dir, 2);
                            let dt = t0.elapsed();
                            // Diagnostic: surface directories that stall the walk.
                            if dt.as_secs_f32() > 1.0 {
                                eprintln!(
                                    "[scan] SLOW {:.1}s {} ({})",
                                    dt.as_secs_f32(),
                                    dir.display(),
                                    match &res {
                                        Ok(l) => format!("{} entries", l.entries.len()),
                                        Err(e) => format!("err: {e}"),
                                    }
                                );
                            }
                            if let Ok(listing) = res {
                                parsed.lock().unwrap().push((dir.clone(), *depth, listing));
                            }
                        });
                    }
                });
                let parsed = parsed.into_inner().unwrap();

                let mut next: Vec<(PathBuf, usize)> = Vec::new();
                for (dir, depth, listing) in parsed {
                    dirs_seen += 1;
                    for entry in &listing.entries {
                        if matches!(entry.kind, EntryKind::Directory)
                            && !entry.hidden
                            && depth < MAX_DEPTH
                            && !skip(&entry.path)
                        {
                            next.push((entry.path.clone(), depth + 1));
                        }
                    }
                    let dir_matches = listing
                        .sequences
                        .iter()
                        .filter(|seq| {
                            walk_needle.is_empty()
                                || seq
                                    .sequence_string()
                                    .map(|n| n.to_lowercase().contains(&walk_needle))
                                    .unwrap_or(false)
                        })
                        .count()
                        + listing
                            .entries
                            .iter()
                            .filter(|e| {
                                matches!(e.kind, EntryKind::File)
                                    && crate::thumbs::is_movie(&e.path)
                                    && (walk_needle.is_empty()
                                        || e.path
                                            .file_name()
                                            .map(|n| {
                                                n.to_string_lossy()
                                                    .to_lowercase()
                                                    .contains(&walk_needle)
                                            })
                                            .unwrap_or(false))
                            })
                            .count();
                    if dir_matches > 0 {
                        matches_atomic.fetch_add(dir_matches, Ordering::Relaxed);
                        found_w.lock().unwrap().push((dir, listing));
                    }
                }
                // Keep any untaken siblings (budget-capped level) plus the new
                // children.
                frontier.extend(next);
            }
        });

    // Race the walk against cancellation AND a hard wall-clock timeout. A
    // worker can wedge inside a blocking filesystem call that no flag can
    // interrupt — e.g. macOS File Provider (Dropbox) hanging on a directory the
    // app lacks TCC permission for, or hydrating an online-only file on stat.
    // The in-loop deadline can't fire while every worker is stuck, so this
    // outer race is the real guarantee: on cancel/timeout we *detach* the walk
    // (the abandoned threads finish on their own; their late results are
    // dropped) and return whatever was found so far.
    //
    // IMPORTANT: no `tokio::time` here. Tauri's async runtime does not
    // reliably drive tokio's timer, so `tokio::time::sleep` inside a command
    // can simply never complete — the timeout/cancel arms hang and the whole
    // scan appears frozen at "0 dirs". A plain OS thread is the watchdog: it
    // polls cancel + deadline + emits rate-limited progress, and signals the
    // command through a oneshot (channels don't need the timer driver).
    let cancel_poll = search.cancel.clone();
    let hard_timeout = std::time::Duration::from_secs(25);
    let walk_done = Arc::new(AtomicBool::new(false));
    let (tx, rx) = tokio::sync::oneshot::channel::<&'static str>();
    {
        let walk_done = walk_done.clone();
        let cur_dir = cur_dir.clone();
        let scanned_atomic = scanned_atomic.clone();
        let matches_shared = matches_shared.clone();
        std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let mut last_dir = String::new();
            let mut last_n = usize::MAX;
            let mut tx = Some(tx);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                // Progress (rate-limited, single sender).
                let dir = cur_dir.lock().map(|s| s.clone()).unwrap_or_default();
                let n = scanned_atomic.load(Ordering::Relaxed);
                if dir != last_dir || n != last_n {
                    last_dir = dir.clone();
                    last_n = n;
                    let _ = progress.send(SearchProgress {
                        dir,
                        scanned: n,
                        matches: matches_shared.load(Ordering::Relaxed),
                    });
                }
                if walk_done.load(Ordering::SeqCst) {
                    break; // walk finished normally; command already returned
                }
                let why = if cancel_poll.load(Ordering::SeqCst) {
                    Some("cancelled")
                } else if start.elapsed() >= hard_timeout {
                    Some("timeout")
                } else {
                    None
                };
                if let Some(why) = why {
                    if let Some(tx) = tx.take() {
                        let _ = tx.send(why);
                    }
                    break;
                }
            }
        });
    }

    // Wait for whichever comes first: the walk finishing, or the watchdog
    // signalling cancel/timeout. On the latter the walk is *detached* — its
    // threads finish on their own; late results are dropped.
    tokio::select! {
        res = walk => {
            walk_done.store(true, Ordering::SeqCst);
            res.map_err(|e| e.to_string())?;
        }
        why = rx => {
            match why {
                Ok("cancelled") => eprintln!("[scan] cancelled by user; walk detached"),
                Ok(_) => eprintln!(
                    "[scan] HARD TIMEOUT after {hard_timeout:?}; walk detached (workers wedged?)"
                ),
                Err(_) => {}
            }
        }
    }
    let matched_listings: Vec<(PathBuf, sequitur::DirectoryListing)> =
        std::mem::take(&mut *found.lock().map_err(|e| e.to_string())?);
    eprintln!(
        "[scan] walk phase done: {} matching dirs; absorbing into registry…",
        matched_listings.len()
    );

    // Absorb each matching directory and keep only the matching entities.
    let t_absorb = std::time::Instant::now();
    let mut registry = state.lock().map_err(|e| e.to_string())?;
    eprintln!("[scan] registry lock acquired ({:?})", t_absorb.elapsed());
    let mut matched_ids: Vec<EntityId> = Vec::new();
    for (dir, listing) in matched_listings {
        let entries = listing
            .entries
            .iter()
            .map(|e| (e.path.clone(), e.kind, e.hidden))
            .collect();
        let outcome = registry.absorb_scan(&dir, listing.sequences, entries);
        for id in outcome.ids {
            let Some(entity) = registry.get(id) else {
                continue;
            };
            let is_match = match &entity.canonical {
                Canonical::Sequence(seq) => {
                    needle.is_empty()
                        || seq
                            .sequence_string()
                            .map(|n| n.to_lowercase().contains(&needle))
                            .unwrap_or(false)
                }
                Canonical::Loose {
                    path,
                    kind: EntryKind::File,
                    ..
                } => {
                    crate::thumbs::is_movie(path)
                        && (needle.is_empty()
                            || path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_lowercase().contains(&needle))
                                .unwrap_or(false))
                }
                _ => false,
            };
            if is_match {
                matched_ids.push(id);
            }
        }
    }
    matched_ids.truncate(MAX_RESULTS);
    eprintln!(
        "[scan] absorbed in {:?}; {} matched ids; computing conflicts…",
        t_absorb.elapsed(),
        matched_ids.len()
    );

    let t_conf = std::time::Instant::now();
    let conflicts = registry.projected_conflicts();
    eprintln!("[scan] conflicts computed in {:?}; building views…", t_conf.elapsed());
    let entities = matched_ids
        .into_iter()
        .filter_map(|id| {
            registry
                .get(id)
                .map(|entity| entity_view(id, entity, conflicts_of(&conflicts, id)))
        })
        .collect();
    eprintln!("[scan] done, returning");

    Ok(DirView {
        path: root.to_string_lossy().into_owned(),
        entities,
        voided: Vec::new(),
    })
}

/// One directory in the sidebar folder tree.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirNode {
    pub name: String,
    pub path: String,
}

/// Non-hidden subdirectories of `path` (symlinks to directories included),
/// sorted by name — the lazy children of one folder-tree node.
#[tauri::command]
pub fn list_dirs(path: String) -> Vec<DirNode> {
    let Ok(entries) = std::fs::read_dir(PathBuf::from(&path)) else {
        return Vec::new();
    };
    let mut out: Vec<DirNode> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || !e.path().is_dir() {
                return None;
            }
            Some(DirNode {
                path: e.path().to_string_lossy().into_owned(),
                name,
            })
        })
        .collect();
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

/// Filesystem roots for the top of the folder tree: `/` on Unix, each present
/// drive (`C:\`, `D:\`, …) on Windows.
#[tauri::command]
pub fn filesystem_roots() -> Vec<DirNode> {
    #[cfg(windows)]
    {
        ('A'..='Z')
            .filter_map(|c| {
                let p = format!("{c}:\\");
                if std::path::Path::new(&p).is_dir() {
                    Some(DirNode {
                        name: format!("{c}:"),
                        path: p,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
    #[cfg(not(windows))]
    {
        vec![DirNode {
            name: "/".to_string(),
            path: "/".to_string(),
        }]
    }
}

/// One playback frame of a sequence as raw JPEG bytes (binary IPC, no JSON
/// overhead): frame `index` into the on-disk frames, scaled to fit `max_dim`
/// (default 1280). Frames are transcoded once per resolution and disk-cached;
/// the frontend keeps its own RAM buffer for realtime playback.
#[tauri::command]
#[allow(clippy::too_many_arguments)] // Tauri commands are flat parameter lists.
pub async fn playback_frame(
    id: EntityId,
    index: usize,
    max_dim: Option<u32>,
    input: Option<crate::thumbs::InputSpace>,
    view: Option<crate::thumbs::ViewTransform>,
    exposure: Option<f32>,
    app: tauri::AppHandle,
    state: State<'_, Mutex<Registry>>,
) -> Result<tauri::ipc::Response, String> {
    use tauri::Manager;
    let color = color_opts(input, view, exposure);

    let path = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        let entity = registry
            .get(id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        match &entity.canonical {
            Canonical::Sequence(seq) => seq
                .items()
                .get(index)
                .map(|item| item.path())
                .ok_or_else(|| format!("no frame at index {index}"))?,
            Canonical::Loose { path, .. } => path.clone(),
        }
    };

    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("playback");
    let max_dim = max_dim.unwrap_or(2048).clamp(64, 4096);

    let bytes = tauri::async_runtime::spawn_blocking(move || {
        crate::thumbs::playback_jpeg(&path, &cache_dir, max_dim, &color)
    })
    .await
    .map_err(|e| e.to_string())??;

    Ok(tauri::ipc::Response::new(bytes))
}

/// Directory candidates for terminal-style tab completion of `partial`.
///
/// `partial` is what's in the path field: everything up to the last `/` is
/// the directory to search, the remainder is the fragment to match (case-
/// insensitively). Only directories (and symlinks to directories) are
/// returned, as absolute paths, sorted. Hidden directories only match when
/// the fragment itself starts with a dot.
#[tauri::command]
pub fn complete_path(partial: String) -> Vec<String> {
    // Split on the last path separator, accepting both `/` and `\` so Windows
    // paths (C:\Users\…) complete as well as POSIX ones.
    let (parent, fragment) = match partial.rfind(['/', '\\']) {
        Some(i) => {
            let dir = &partial[..i];
            let frag = partial[i + 1..].to_string();
            // "" -> POSIX root; "C:" -> the drive's root, not its CWD.
            let dir = if dir.is_empty() {
                "/".to_string()
            } else if dir.ends_with(':') {
                format!("{dir}\\")
            } else {
                dir.to_string()
            };
            (PathBuf::from(dir), frag)
        }
        None => return Vec::new(),
    };
    let show_hidden = fragment.starts_with('.');
    let fragment_lower = fragment.to_lowercase();

    let Ok(entries) = std::fs::read_dir(&parent) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.to_lowercase().starts_with(&fragment_lower) {
                return None;
            }
            if name.starts_with('.') && !show_hidden {
                return None;
            }
            // `is_dir` follows symlinks, so a link to a directory completes.
            if !e.path().is_dir() {
                return None;
            }
            Some(parent.join(name).to_string_lossy().into_owned())
        })
        .collect();
    out.sort();
    out
}

/// A label/value row in the viewer's info panel.
#[derive(Debug, Serialize)]
pub struct InfoField {
    pub label: String,
    pub value: String,
}

/// Technical media info for the viewer: sequence range/size plus per-frame
/// resolution/format and format-specific fields (EXR channels/compression,
/// movie codec/fps, etc.).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaInfo {
    pub name: String,
    pub path: String,
    pub format: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fields: Vec<InfoField>,
}

fn field(label: &str, value: impl Into<String>) -> InfoField {
    InfoField {
        label: label.to_string(),
        value: value.into(),
    }
}

fn human_bytes(n: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < units.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", units[i])
    }
}

/// Probes an entity's media and returns viewer-facing technical info.
#[tauri::command]
pub async fn media_info(
    id: EntityId,
    state: State<'_, Mutex<Registry>>,
) -> Result<MediaInfo, String> {
    // Snapshot what we need under the lock.
    enum Probe {
        Sequence {
            name: String,
            paths: Vec<PathBuf>,
            first: i32,
            last: i32,
            span: i32,
            present: usize,
            missing: usize,
        },
        File {
            name: String,
            path: PathBuf,
        },
    }
    let probe = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        let entity = registry
            .get(id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        match &entity.canonical {
            Canonical::Sequence(seq) => Probe::Sequence {
                name: seq
                    .sequence_string()
                    .unwrap_or_else(|_| "<sequence>".to_string()),
                paths: seq.items().iter().map(|i| i.path()).collect(),
                first: seq.first_frame(),
                last: seq.last_frame(),
                span: seq.frame_count(),
                present: seq.actual_frame_count(),
                missing: seq.missing_frame_count() as usize,
            },
            Canonical::Loose { path, .. } => Probe::File {
                name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                path: path.clone(),
            },
        }
    };

    tauri::async_runtime::spawn_blocking(move || {
        let mut fields = Vec::new();
        let (name, rep_path) = match &probe {
            Probe::Sequence {
                name,
                paths,
                first,
                last,
                span,
                present,
                missing,
            } => {
                fields.push(field("Frame range", format!("{first}–{last}")));
                fields.push(field("Frames", format!("{present} / {span}")));
                if *missing > 0 {
                    fields.push(field("Missing", missing.to_string()));
                }
                let total: u64 = paths
                    .iter()
                    .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                    .sum();
                fields.push(field("Total size", human_bytes(total)));
                let rep = crate::thumbs::representative_path(paths)
                    .unwrap_or_else(|| paths[0].clone());
                (name.clone(), rep)
            }
            Probe::File { name, path } => {
                if let Ok(m) = std::fs::metadata(path) {
                    fields.push(field("Size", human_bytes(m.len())));
                }
                (name.clone(), path.clone())
            }
        };

        let (width, height) = probe_technical(&rep_path, &mut fields);
        MediaInfo {
            name,
            path: rep_path.to_string_lossy().into_owned(),
            format: format_label(&rep_path),
            width,
            height,
            fields,
        }
    })
    .await
    .map_err(|e| e.to_string())
}

fn format_label(path: &Path) -> String {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("exr") => "OpenEXR".to_string(),
        Some("dpx") => "DPX".to_string(),
        Some("jpg" | "jpeg") => "JPEG".to_string(),
        Some("png") => "PNG".to_string(),
        Some("tif" | "tiff") => "TIFF".to_string(),
        Some("mov") => "QuickTime".to_string(),
        Some("mp4" | "m4v") => "MP4".to_string(),
        Some(other) => other.to_uppercase(),
        None => "—".to_string(),
    }
}

/// Per-format technical probe; appends fields and returns (width, height).
fn probe_technical(path: &Path, fields: &mut Vec<InfoField>) -> (Option<u32>, Option<u32>) {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    if ext == "exr" {
        if let Ok((w, h)) = probe_exr(path, fields) {
            fields.insert(0, field("Resolution", format!("{w} × {h}")));
            return (Some(w), Some(h));
        }
    } else if crate::thumbs::is_movie(path) || ext == "dpx" {
        if let Some((w, h)) = probe_ffmpeg(path, fields) {
            fields.insert(0, field("Resolution", format!("{w} × {h}")));
            return (Some(w), Some(h));
        }
    } else if let Ok((w, h)) = image::image_dimensions(path) {
        fields.insert(0, field("Resolution", format!("{w} × {h}")));
        return (Some(w), Some(h));
    }
    (None, None)
}

fn probe_exr(path: &Path, fields: &mut Vec<InfoField>) -> Result<(u32, u32), String> {
    let meta = exr::meta::MetaData::read_from_file(path, false).map_err(|e| e.to_string())?;
    let header = meta.headers.first().ok_or("no exr header")?;
    let size = header.layer_size;
    let channels: Vec<String> = header
        .channels
        .list
        .iter()
        .map(|c| c.name.to_string())
        .collect();
    let depth = header
        .channels
        .list
        .first()
        .map(|c| format!("{:?}", c.sample_type))
        .unwrap_or_default();
    if !channels.is_empty() {
        fields.push(field("Channels", channels.join(", ")));
    }
    if !depth.is_empty() {
        fields.push(field("Bit depth", depth));
    }
    fields.push(field("Compression", format!("{:?}", header.compression)));
    Ok((size.width() as u32, size.height() as u32))
}

/// Parses ffmpeg's `-i` banner for the first video stream's resolution,
/// codec, pixel format, fps and the clip duration.
fn probe_ffmpeg(path: &Path, fields: &mut Vec<InfoField>) -> Option<(u32, u32)> {
    let ffmpeg = crate::thumbs::ffmpeg_path()?;
    let out = crate::thumbs::hidden_command(ffmpeg)
        .arg("-i")
        .arg(path)
        .output()
        .ok()?;
    let banner = String::from_utf8_lossy(&out.stderr);

    if let Some(line) = banner.lines().find(|l| l.contains("Stream") && l.contains("Video:")) {
        let after = line.split("Video:").nth(1).unwrap_or("").trim();
        if let Some(codec) = after.split([' ', ',']).find(|s| !s.is_empty()) {
            fields.push(field("Codec", codec.to_string()));
        }
        // resolution like 1920x1080
        let mut res = None;
        for tok in after.split([' ', ',', '(', ')']) {
            if let Some((w, h)) = tok.split_once('x') {
                if let (Ok(w), Ok(h)) = (w.trim().parse::<u32>(), h.trim().parse::<u32>()) {
                    res = Some((w, h));
                    break;
                }
            }
        }
        if let Some(idx) = after.find(" fps") {
            let pre: String = after[..idx]
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let fps: String = pre.chars().rev().collect();
            if !fps.is_empty() {
                fields.push(field("FPS", fps));
            }
        }
        if let Some(dl) = banner.lines().find(|l| l.trim_start().starts_with("Duration:")) {
            if let Some(d) = dl.split("Duration:").nth(1).and_then(|s| s.split(',').next()) {
                fields.push(field("Duration", d.trim().to_string()));
            }
        }
        return res;
    }
    None
}

/// Filesystem metadata for one entity, summed across a sequence's frames.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetaDto {
    pub id: EntityId,
    /// Total bytes across all frames (None for directories — not summed).
    pub total_size: Option<u64>,
    /// Latest mtime across frames, Unix seconds.
    pub mtime: Option<i64>,
    /// Owning user name (uid string if it can't be resolved).
    pub owner: Option<String>,
}

/// Stats the on-disk files behind `ids` and returns size/mtime/owner. A
/// separate command from the listing so the directory view stays fast — the
/// frontend fetches this lazily and fills in the columns when it arrives.
/// One stat pass per file; sequences sum size and take the newest mtime.
#[tauri::command]
pub async fn entity_metadata(
    ids: Vec<EntityId>,
    state: State<'_, Mutex<Registry>>,
) -> Result<Vec<MetaDto>, String> {
    // (id, frame paths, is_directory) snapshot under the lock.
    let jobs: Vec<(EntityId, Vec<PathBuf>, bool)> = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        ids.into_iter()
            .filter_map(|id| {
                let entity = registry.get(id)?;
                let (paths, is_dir) = match &entity.canonical {
                    Canonical::Sequence(seq) => {
                        (seq.items().iter().map(|i| i.path()).collect(), false)
                    }
                    Canonical::Loose { path, kind, .. } => (
                        vec![path.clone()],
                        matches!(kind, EntryKind::Directory),
                    ),
                };
                Some((id, paths, is_dir))
            })
            .collect()
    };

    tauri::async_runtime::spawn_blocking(move || {
        jobs.into_iter()
            .map(|(id, paths, is_dir)| metadata_for(id, &paths, is_dir))
            .collect()
    })
    .await
    .map_err(|e| e.to_string())
}

fn metadata_for(id: EntityId, paths: &[PathBuf], is_dir: bool) -> MetaDto {
    let mut total: u64 = 0;
    let mut mtime: Option<i64> = None;
    let mut owner: Option<String> = None;
    let mut any = false;

    for path in paths {
        let Ok(meta) = std::fs::metadata(path) else {
            continue;
        };
        any = true;
        total = total.saturating_add(meta.len());
        if let Ok(modified) = meta.modified() {
            if let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
                let secs = dur.as_secs() as i64;
                mtime = Some(mtime.map_or(secs, |m: i64| m.max(secs)));
            }
        }
        if owner.is_none() {
            owner = owner_of(&meta);
        }
    }

    MetaDto {
        id,
        total_size: if is_dir || !any { None } else { Some(total) },
        mtime,
        owner,
    }
}

#[cfg(unix)]
fn owner_of(meta: &std::fs::Metadata) -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let uid = meta.uid();
    Some(
        uzers::get_user_by_uid(uid)
            .map(|u| u.name().to_string_lossy().into_owned())
            .unwrap_or_else(|| uid.to_string()),
    )
}

#[cfg(not(unix))]
fn owner_of(_meta: &std::fs::Metadata) -> Option<String> {
    None
}

/// What a commit would destroy unrecoverably.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermanentDeletes {
    /// Files that would be deleted with no trash to recover them from.
    pub count: usize,
    /// The distinct directories holding them, to name in the warning.
    pub dirs: Vec<String>,
}

/// Inspects what committing `ids` (or everything staged) would delete, and
/// reports the part that cannot be undone — Windows keeps no Recycle Bin on
/// network shares.
///
/// Read off the real commit plan rather than the intent log, so it counts the
/// frames a trim drops as well as whole-entity deletes, and mirrors the same
/// entity selection and broken-entity skipping as [`run_commit`].
///
/// Trash availability is a property of the volume, so it's resolved once per
/// directory: probing 1000 frames individually would mean 1000 round trips to
/// the share.
#[tauri::command]
pub fn permanent_deletes(
    ids: Option<Vec<EntityId>>,
    state: AppState,
) -> Result<PermanentDeletes, String> {
    let registry = state.lock().map_err(|e| e.to_string())?;
    let mut checked: HashMap<PathBuf, bool> = HashMap::new();
    let mut count = 0usize;
    let mut dirs: Vec<String> = Vec::new();
    for (id, entity) in registry.pending() {
        if ids.as_ref().is_some_and(|ids| !ids.contains(&id)) {
            continue;
        }
        let Ok(plan) = entity.commit_plan() else {
            continue;
        };
        for op in plan.operations() {
            let FileOperation::Delete { source } = op else {
                continue;
            };
            let dir = source.parent().unwrap_or(source).to_path_buf();
            let recoverable = *checked
                .entry(dir.clone())
                .or_insert_with(|| sequitur::trash_available(&dir));
            if recoverable {
                continue;
            }
            count += 1;
            let shown = dir.to_string_lossy().into_owned();
            if !dirs.contains(&shown) {
                dirs.push(shown);
            }
        }
    }
    Ok(PermanentDeletes { count, dirs })
}

/// What executing the delete queue would remove.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeletePreview {
    /// Individual files on disk (a sequence counts every frame).
    pub files: usize,
    /// Their total size in bytes.
    pub bytes: u64,
    /// Distinct directories with no trash to recover from — a delete there is
    /// permanent even when the user asked for trash.
    pub no_trash_dirs: Vec<String>,
}

/// Files behind `ids`, flattened: every frame of a sequence, the path of a
/// loose file. Directories are refused — the queue only takes media.
fn files_of(registry: &Registry, ids: &[EntityId]) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for id in ids {
        let Some(entity) = registry.get(*id) else {
            continue; // already gone from disk; silently skipped
        };
        match &entity.canonical {
            Canonical::Sequence(seq) => out.extend(seq.items().iter().map(|i| i.path())),
            Canonical::Loose {
                path,
                kind: EntryKind::File,
                ..
            } => out.push(path.clone()),
            Canonical::Loose { kind, .. } => {
                return Err(format!("{kind:?} entries cannot be queued for delete"))
            }
        }
    }
    Ok(out)
}

/// Totals for the delete-queue review page: how many files, how big, and
/// which directories can't offer a recoverable delete.
///
/// Trash availability is a property of the volume, so it is probed once per
/// directory rather than per file.
#[tauri::command]
pub fn delete_preview(ids: Vec<EntityId>, state: AppState) -> Result<DeletePreview, String> {
    let registry = state.lock().map_err(|e| e.to_string())?;
    let paths = files_of(&registry, &ids)?;
    let mut checked: HashMap<PathBuf, bool> = HashMap::new();
    let mut bytes = 0u64;
    let mut no_trash_dirs: Vec<String> = Vec::new();
    for p in &paths {
        bytes += fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let dir = p.parent().unwrap_or(p).to_path_buf();
        let recoverable = *checked
            .entry(dir.clone())
            .or_insert_with(|| sequitur::trash_available(&dir));
        if !recoverable {
            let shown = dir.to_string_lossy().into_owned();
            if !no_trash_dirs.contains(&shown) {
                no_trash_dirs.push(shown);
            }
        }
    }
    Ok(DeletePreview {
        files: paths.len(),
        bytes,
        no_trash_dirs,
    })
}

/// Outcome of executing the delete queue.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteOutcome {
    /// Files actually removed.
    pub deleted: usize,
    /// One message per file that could not be removed; the rest still went.
    pub failed: Vec<String>,
}

/// Deletes every file behind `ids`, to the trash when `to_trash` (and the
/// volume has one), otherwise permanently.
///
/// This is deliberately NOT the stage/commit path: the queue is its own
/// review-then-execute flow, and deletes are order-independent so they need
/// none of the executor's cycle-safe scheduling. A failure on one file is
/// reported but does not abort the rest — a half-finished delete run is worse
/// than a fully-reported one. The affected directories are re-scanned
/// afterwards so the registry drops the entities that are now gone.
#[tauri::command]
pub async fn delete_entities(
    ids: Vec<EntityId>,
    to_trash: bool,
    progress: Channel<ProgressDto>,
    state: AppState<'_>,
    commit: State<'_, CommitState>,
) -> Result<DeleteOutcome, String> {
    if commit.in_flight.load(Ordering::SeqCst) {
        return Err(COMMIT_BUSY.to_string());
    }
    let (paths, dirs) = {
        let registry = state.lock().map_err(|e| e.to_string())?;
        let paths = files_of(&registry, &ids)?;
        let mut dirs: Vec<PathBuf> = paths
            .iter()
            .map(|p| p.parent().unwrap_or(p).to_path_buf())
            .collect();
        dirs.sort();
        dirs.dedup();
        (paths, dirs)
    };
    if paths.is_empty() {
        return Ok(DeleteOutcome {
            deleted: 0,
            failed: Vec::new(),
        });
    }

    let total = paths.len();
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let mut deleted = 0usize;
        let mut failed = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            match sequitur::delete_file(p, to_trash) {
                Ok(()) => deleted += 1,
                Err(e) => failed.push(format!("{}: {e}", p.display())),
            }
            let _ = progress.send(ProgressDto {
                index: i + 1,
                total,
                label: p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            });
        }
        DeleteOutcome { deleted, failed }
    })
    .await
    .map_err(|e| e.to_string())?;

    // Re-scan so the registry forgets what is no longer on disk.
    let mut registry = state.lock().map_err(|e| e.to_string())?;
    for dir in dirs {
        if let Ok(listing) = FileSequence::parse_directory(&dir, 2) {
            let entries = listing
                .entries
                .iter()
                .map(|e| (e.path.clone(), e.kind, e.hidden))
                .collect();
            registry.absorb_scan(&dir, listing.sequences, entries);
        }
    }
    Ok(outcome)
}

/// Returns the user's home directory as a sensible default starting point.
#[tauri::command]
pub fn home_directory() -> Result<String, String> {
    home_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| "could not determine home directory".to_string())
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty() && Path::new(p).is_dir())
}

/// Hand sample frame paths to a running wobble edit instance over its
/// Unix-socket ingest (one JSON line per sequence; wobble edit adds each as
/// a timeline clip). Errors if wobble edit isn't listening.
#[tauri::command]
pub fn send_to_wobble_edit(samples: Vec<String>) -> Result<usize, String> {
    let sock = home_dir()
        .ok_or_else(|| "could not determine home directory".to_string())?
        .join("Library/Application Support/WobbleEdit/ingest.sock");
    #[cfg(unix)]
    {
        use std::io::Write as _;
        let mut conn = std::os::unix::net::UnixStream::connect(&sock)
            .map_err(|_| "wobble edit isn't running (no ingest socket)".to_string())?;
        for s in &samples {
            let line = serde_json::json!({ "sample": s }).to_string() + "\n";
            conn.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(samples.len())
    }
    // No Unix sockets: the ingest is not offered here.
    #[cfg(not(unix))]
    {
        let _ = (sock, samples);
        Err("sending to wobble edit needs a Unix socket (not on this platform)".to_string())
    }
}
