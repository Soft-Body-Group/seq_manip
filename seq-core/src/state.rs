//! Rust-canonical app state: the entity registry, staged intents, and the
//! pure projection fold.
//!
//! Svelte is a projection renderer; it addresses entities by `EntityId` over
//! IPC and never holds paths as identity. Each entity pairs an on-disk
//! snapshot (`Canonical`) with an ordered log of staged edits (`IntentOp`);
//! the projected appearance is `fold(canonical, intents)` — pure, no IO.

use sequitur::{
    Components, EntryKind, FileOperation, FileSequence, OperationPlan, SequenceError,
};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub type EntityId = u64;

/// The on-disk snapshot an entity was last seen as.
#[derive(Debug, Clone)]
pub enum Canonical {
    Sequence(FileSequence),
    Loose {
        path: PathBuf,
        kind: EntryKind,
        hidden: bool,
    },
}

/// Optional new-name components for a staged rename; unset fields keep the
/// entity's existing component (mirrors sequitur's `Components` builder).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NameComponents {
    pub prefix: Option<String>,
    pub delimiter: Option<String>,
    pub suffix: Option<String>,
    pub extension: Option<String>,
    pub padding: Option<usize>,
}

impl NameComponents {
    fn to_components(&self) -> Components {
        let mut c = Components::new();
        if let Some(v) = &self.prefix {
            c = c.prefix(v);
        }
        if let Some(v) = &self.delimiter {
            c = c.delimiter(v);
        }
        if let Some(v) = &self.suffix {
            c = c.suffix(v);
        }
        if let Some(v) = &self.extension {
            c = c.extension(v);
        }
        if let Some(v) = self.padding {
            c = c.padding(v);
        }
        c
    }
}

/// The op-log alphabet: user intent, not pre-baked plans. Stored per entity
/// and replayable against a fresh snapshot (rebase) after external changes.
///
/// Generative `Copy` is not an `IntentOp`: it mints a new entity (see
/// [`Registry::stage_copy`]) whose log then holds ordinary transforms.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum IntentOp {
    /// Component-wise rename; sequences only.
    Rename { to: NameComponents },
    /// Whole-filename rename; loose files and directories only.
    RenameFile { name: String },
    Move { dest: PathBuf },
    Delete,
    Offset { by: i32, padding: Option<usize> },
    Repad { width: usize },
    /// Keep only frames in the inclusive range `first..=last`; sequences only.
    /// The frames outside it are deleted at commit — this is the one verb that
    /// changes an entity's frame COUNT, which [`Entity::commit_plan`] has to
    /// account for when pairing canonical items with projected ones.
    /// Surviving frames keep their numbers (renumbering is `Offset`'s job).
    Trim { first: i32, last: i32 },
}

impl IntentOp {
    /// Human-readable one-liner for the pending-changes tray.
    pub fn summary(&self) -> String {
        match self {
            Self::Rename { .. } => "rename".to_string(),
            Self::RenameFile { name } => format!("rename to {name}"),
            Self::Move { dest } => format!("move to {}", dest.display()),
            Self::Delete => "delete".to_string(),
            Self::Offset { by, .. } => format!("offset frames by {by:+}"),
            Self::Repad { width } => format!("repad to {width} digits"),
            Self::Trim { first, last } => format!("trim to frames {first}-{last}"),
        }
    }
}

/// The result of folding an entity's intents over its canonical snapshot.
#[derive(Debug, Clone)]
pub struct Projection {
    pub projected: Projected,
    pub deleted: bool,
    /// Canonical paths dropped by a staged [`IntentOp::Trim`]: they exist on
    /// disk now but not in `projected`, so commit deletes them. Kept separate
    /// from `deleted` (which retires the whole entity) and always in canonical
    /// terms, so it survives later renames/offsets in the same log.
    pub trimmed: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum Projected {
    Sequence(FileSequence),
    Loose(PathBuf),
}

/// How an entity's staged state relates to disk. Mirrors the architecture
/// doc's lifecycle enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Unchanged,
    Modified,
    Deleted,
    Departing,
    /// A staged copy: nothing on disk yet; commit creates it.
    New,
}

impl Lifecycle {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unchanged => "unchanged",
            Self::Modified => "modified",
            Self::Deleted => "deleted",
            Self::Departing => "departing",
            Self::New => "new",
        }
    }
}

/// Where an entity's canonical snapshot came from.
///
/// `Disk` entities mirror something that exists; their commit plans rename in
/// place. A `CopyOf` entity is *generative*: its canonical snapshot is a clone
/// of the source's (the paths it will copy **from**), its intent log holds the
/// transforms that differentiate the copy (new name and/or directory), and its
/// commit plan emits `Copy` operations. On a successful commit it becomes a
/// `Disk` entity; discarding it just removes it (nothing on disk to undo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Disk,
    CopyOf(EntityId),
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub canonical: Canonical,
    pub intents: Vec<IntentOp>,
    pub origin: Origin,
}

impl Entity {
    /// The directory this entity canonically lives in.
    pub fn canonical_dir(&self) -> Option<PathBuf> {
        match &self.canonical {
            Canonical::Sequence(seq) => seq.directory().ok().flatten().cloned(),
            Canonical::Loose { path, .. } => path.parent().map(Path::to_path_buf),
        }
    }

    /// Display name of the canonical (pre-edit) state.
    pub fn canonical_name(&self) -> String {
        match &self.canonical {
            Canonical::Sequence(seq) => seq
                .sequence_string()
                .unwrap_or_else(|_| "<sequence>".to_string()),
            Canonical::Loose { path, .. } => path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        }
    }

    /// Pure fold of the intent log over the canonical snapshot. No IO: each
    /// sequitur verb returns `Planned { proposed, .. }` and we thread
    /// `proposed` through, discarding the per-verb plans (the commit plan is
    /// built later as a minimal diff, not by concatenation).
    pub fn project(&self) -> Result<Projection, SequenceError> {
        let mut deleted = false;
        let mut trimmed = Vec::new();
        match &self.canonical {
            Canonical::Sequence(seq) => {
                let mut cur = seq.clone();
                // The canonical items still standing, held in lockstep with
                // `cur` so a trim can cut the same positions from both. Every
                // other verb is order- and count-preserving, which is what lets
                // commit_plan pair the two positionally afterwards.
                let mut surviving = seq.clone();
                for intent in &self.intents {
                    match intent {
                        IntentOp::Rename { to } => cur = cur.rename(to.to_components()).proposed,
                        IntentOp::Move { dest } => cur = cur.move_to(dest).proposed,
                        IntentOp::Offset { by, padding } => {
                            cur = cur.offset_frames(*by, *padding)?.proposed
                        }
                        IntentOp::Repad { width } => cur = cur.with_padding(*width).proposed,
                        IntentOp::Delete => deleted = true,
                        IntentOp::Trim { first, last } => {
                            // Range applies to the frame numbers as projected
                            // *at this point* in the log, so a trim staged after
                            // an offset means what the user saw when staging it.
                            let mut kept = Vec::new();
                            let mut kept_canonical = Vec::new();
                            for (projected, canonical) in
                                cur.items().iter().zip(surviving.items())
                            {
                                if (*first..=*last).contains(&projected.frame_number()) {
                                    kept.push(projected.clone());
                                    kept_canonical.push(canonical.clone());
                                } else {
                                    trimmed.push(canonical.path());
                                }
                            }
                            // An empty result is EmptySequence, which stage_intent
                            // turns into a rejection rather than a broken entity.
                            cur = FileSequence::new(kept)?;
                            surviving = FileSequence::new(kept_canonical)?;
                        }
                        // stage_intent rejects this on sequences.
                        IntentOp::RenameFile { .. } => {}
                    }
                }
                Ok(Projection {
                    projected: Projected::Sequence(cur),
                    deleted,
                    trimmed,
                })
            }
            Canonical::Loose { path, .. } => {
                let mut cur = path.clone();
                for intent in &self.intents {
                    match intent {
                        IntentOp::Move { dest } => {
                            if let Some(name) = cur.file_name() {
                                cur = dest.join(name);
                            }
                        }
                        IntentOp::RenameFile { name } => {
                            cur = cur
                                .parent()
                                .map(|p| p.join(name))
                                .unwrap_or_else(|| PathBuf::from(name));
                        }
                        IntentOp::Delete => deleted = true,
                        // stage_intent rejects these on loose entities.
                        IntentOp::Rename { .. }
                        | IntentOp::Offset { .. }
                        | IntentOp::Repad { .. }
                        | IntentOp::Trim { .. } => {}
                    }
                }
                Ok(Projection {
                    projected: Projected::Loose(cur),
                    deleted,
                    trimmed,
                })
            }
        }
    }

    /// Whether this is an uncommitted generative copy.
    pub fn is_copy(&self) -> bool {
        matches!(self.origin, Origin::CopyOf(_))
    }

    /// Lifecycle derived from the projection relative to the canonical state.
    pub fn lifecycle(&self, projection: &Projection) -> Lifecycle {
        if self.is_copy() {
            return Lifecycle::New;
        }
        if projection.deleted {
            return Lifecycle::Deleted;
        }
        if self.intents.is_empty() {
            return Lifecycle::Unchanged;
        }
        let projected_dir = match &projection.projected {
            Projected::Sequence(seq) => seq.directory().ok().flatten().cloned(),
            Projected::Loose(path) => path.parent().map(Path::to_path_buf),
        };
        if projected_dir != self.canonical_dir() {
            Lifecycle::Departing
        } else {
            Lifecycle::Modified
        }
    }

    /// Every path this entity will occupy once its staged intents are
    /// committed. A staged delete occupies nothing.
    pub fn projected_paths(&self) -> Result<Vec<PathBuf>, SequenceError> {
        let projection = self.project()?;
        if projection.deleted {
            return Ok(Vec::new());
        }
        Ok(match &projection.projected {
            Projected::Sequence(seq) => seq.items().iter().map(|item| item.path()).collect(),
            Projected::Loose(path) => vec![path.clone()],
        })
    }

    /// The minimal set of file operations that takes this entity from its
    /// canonical (on-disk) state to its projected state.
    ///
    /// NOT a concatenation of per-intent plans: the transform stack is
    /// coalesced to its end-state, then canonical and projected items are
    /// zipped pairwise (the fold preserves frame order) and one rename is
    /// emitted per item whose path actually changed. Cancelling edits diff
    /// to nothing. A staged delete emits one remove per item instead.
    pub fn commit_plan(&self) -> Result<OperationPlan, String> {
        let projection = self.project().map_err(|e| e.to_string())?;
        let mut plan = OperationPlan::new();
        // A generative copy reads its canonical (source) paths and writes its
        // projected paths; pairs that didn't diverge are skipped (they'd be
        // a self-copy, and the conflict gate flags them anyway).
        let is_copy = self.is_copy();
        match (&self.canonical, &projection.projected) {
            (Canonical::Sequence(canon), Projected::Sequence(proj)) => {
                if projection.deleted {
                    for item in canon.items() {
                        plan.push(FileOperation::Delete {
                            source: item.path(),
                        });
                    }
                } else {
                    // Frames a trim dropped: gone from `proj`, so they must be
                    // filtered out of the canonical side too or the pairing
                    // below skews by one per dropped frame. A *copy* trims by
                    // simply not copying them — the files are the original's,
                    // so only a real edit deletes them.
                    let dropped: HashSet<PathBuf> = projection.trimmed.iter().cloned().collect();
                    if !is_copy {
                        for source in &projection.trimmed {
                            plan.push(FileOperation::Delete {
                                source: source.clone(),
                            });
                        }
                    }
                    let survivors = canon.items().iter().filter(|i| !dropped.contains(&i.path()));
                    for (old, new) in survivors.zip(proj.items()) {
                        let (source, destination) = (old.path(), new.path());
                        if source != destination {
                            plan.push(if is_copy {
                                FileOperation::Copy {
                                    source,
                                    destination,
                                }
                            } else {
                                FileOperation::Rename {
                                    source,
                                    destination,
                                }
                            });
                        }
                    }
                }
            }
            (Canonical::Loose { path, kind, .. }, Projected::Loose(new_path)) => {
                if projection.deleted {
                    if !matches!(kind, EntryKind::File) {
                        return Err("only plain files can be deleted for now".to_string());
                    }
                    plan.push(FileOperation::Delete {
                        source: path.clone(),
                    });
                } else if path != new_path {
                    plan.push(if is_copy {
                        FileOperation::Copy {
                            source: path.clone(),
                            destination: new_path.clone(),
                        }
                    } else {
                        FileOperation::Rename {
                            source: path.clone(),
                            destination: new_path.clone(),
                        }
                    });
                }
            }
            _ => unreachable!("projection kind always matches canonical kind"),
        }
        Ok(plan)
    }
}

/// Session-stable grouping key for a sequence: identity survives re-scans as
/// long as the group matches. Padding is part of the key because sequitur
/// splits anomalous padding into *separate* `FileSequence`s that share every
/// other component (e.g. `plate_001..099` and `plate_0100`); without padding
/// here those split fragments would collide to one key, overwrite each other
/// in the registry index, and transplant intents onto the wrong fragment.
fn sequence_key(seq: &FileSequence) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}",
        seq.directory()
            .ok()
            .flatten()
            .map(|d| d.display().to_string())
            .unwrap_or_default(),
        seq.prefix().unwrap_or_default(),
        seq.delimiter().ok().flatten().unwrap_or_default(),
        seq.suffix().ok().flatten().unwrap_or_default(),
        seq.extension().unwrap_or_default(),
        seq.nominal_padding(),
    )
}

/// What a directory scan produced: the entities now in the directory, plus
/// the names of entities whose staged edits were voided by external changes.
#[derive(Debug, Default)]
pub struct ScanOutcome {
    pub ids: Vec<EntityId>,
    pub voided: Vec<String>,
}

/// The session registry: every entity ever seen, keyed by a stable id that
/// gestures address (never paths).
#[derive(Debug, Default)]
pub struct Registry {
    next_id: EntityId,
    entities: HashMap<EntityId, Entity>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// All entities with staged intents (the pending-changes tray).
    pub fn pending(&self) -> Vec<(EntityId, &Entity)> {
        let mut out: Vec<(EntityId, &Entity)> = self
            .entities
            .iter()
            .filter(|(_, e)| !e.intents.is_empty())
            .map(|(id, e)| (*id, e))
            .collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    /// Absorbs a fresh scan of `dir`, reconciling against what the registry
    /// already knows so identity (and staged intents) survive re-scans:
    /// sequences match by grouping key, loose entries by exact path; matched
    /// entities get their canonical snapshot refreshed, unmatched scan results
    /// mint new ids, and entities of `dir` that vanished from disk are
    /// dropped (their staged intents are void — the target is gone).
    ///
    /// Returns the ids of the entities now canonically in `dir` (sequences
    /// first, each group in scan order) plus the display names of dropped
    /// entities that still had staged intents, so the UI can tell the user
    /// their edits were voided by an external change.
    pub fn absorb_scan(
        &mut self,
        dir: &Path,
        sequences: Vec<FileSequence>,
        entries: Vec<(PathBuf, EntryKind, bool)>,
    ) -> ScanOutcome {
        // Index the registry's current occupants of this directory. Copies
        // are virtual (their canonical paths are the SOURCE's, nothing of
        // theirs is on disk yet) so they take no part in reconciliation.
        let mut seq_index: HashMap<String, EntityId> = HashMap::new();
        let mut loose_index: HashMap<PathBuf, EntityId> = HashMap::new();
        for (id, entity) in &self.entities {
            if entity.is_copy() || entity.canonical_dir().as_deref() != Some(dir) {
                continue;
            }
            match &entity.canonical {
                Canonical::Sequence(seq) => {
                    seq_index.insert(sequence_key(seq), *id);
                }
                Canonical::Loose { path, .. } => {
                    loose_index.insert(path.clone(), *id);
                }
            }
        }

        let mut ids = Vec::new();
        let mut survivors: Vec<EntityId> = Vec::new();

        for seq in sequences {
            let key = sequence_key(&seq);
            let id = match seq_index.get(&key) {
                Some(&id) => {
                    // Refresh the snapshot; staged intents replay onto it.
                    if let Some(e) = self.entities.get_mut(&id) {
                        e.canonical = Canonical::Sequence(seq);
                    }
                    id
                }
                None => self.insert(Canonical::Sequence(seq)),
            };
            survivors.push(id);
            ids.push(id);
        }

        for (path, kind, hidden) in entries {
            let id = match loose_index.get(&path) {
                Some(&id) => {
                    if let Some(e) = self.entities.get_mut(&id) {
                        e.canonical = Canonical::Loose { path, kind, hidden };
                    }
                    id
                }
                None => self.insert(Canonical::Loose { path, kind, hidden }),
            };
            survivors.push(id);
            ids.push(id);
        }

        // Anything of this dir the scan no longer sees is gone from disk
        // (copies exempt: they have nothing on disk to see). Dropped entities
        // that still carried staged intents are reported as voided.
        let stale: Vec<EntityId> = self
            .entities
            .iter()
            .filter(|(id, e)| {
                !e.is_copy()
                    && e.canonical_dir().as_deref() == Some(dir)
                    && !survivors.contains(id)
            })
            .map(|(id, _)| *id)
            .collect();
        let mut voided = Vec::new();
        for id in stale {
            if let Some(entity) = self.entities.remove(&id) {
                if !entity.intents.is_empty() {
                    voided.push(entity.canonical_name());
                }
            }
        }

        ScanOutcome { ids, voided }
    }

    /// Uncommitted copies whose *projected* home is `dir` — appended to a
    /// directory view so a staged copy is visible where it will land.
    pub fn copies_in(&self, dir: &Path) -> Vec<EntityId> {
        let mut out: Vec<EntityId> = self
            .entities
            .iter()
            .filter(|(_, e)| e.is_copy())
            .filter(|(_, e)| {
                e.project().is_ok_and(|p| {
                    let projected_dir = match &p.projected {
                        Projected::Sequence(seq) => seq.directory().ok().flatten().cloned(),
                        Projected::Loose(path) => path.parent().map(Path::to_path_buf),
                    };
                    projected_dir.as_deref() == Some(dir)
                })
            })
            .map(|(id, _)| *id)
            .collect();
        out.sort();
        out
    }

    fn insert(&mut self, canonical: Canonical) -> EntityId {
        self.insert_with_origin(canonical, Origin::Disk)
    }

    fn insert_with_origin(&mut self, canonical: Canonical, origin: Origin) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                canonical,
                intents: Vec::new(),
                origin,
            },
        );
        id
    }

    /// Stages a generative copy of `id`: mints a new entity whose canonical
    /// snapshot is the source's (the paths to copy from) and whose intent log
    /// holds the differentiating transforms — `prefix` renames the copy
    /// (sequences only), `name` renames it (loose files only), `dest` moves
    /// it. At least one is required, otherwise the copy would land exactly on
    /// its source. The source is untouched.
    pub fn stage_copy(
        &mut self,
        id: EntityId,
        dest: Option<PathBuf>,
        prefix: Option<String>,
        name: Option<String>,
    ) -> Result<EntityId, String> {
        let source = self
            .entities
            .get(&id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        if source.is_copy() {
            return Err("commit the copy before copying it again".to_string());
        }
        match &source.canonical {
            Canonical::Loose {
                kind: EntryKind::Directory,
                ..
            } => return Err("directory copy is not supported yet".to_string()),
            Canonical::Loose {
                kind: EntryKind::Symlink | EntryKind::Other,
                ..
            } => return Err("symlinks and special files cannot be copied".to_string()),
            _ => {}
        }
        if dest.is_none() && prefix.is_none() && name.is_none() {
            return Err("a copy needs a new name or a destination".to_string());
        }
        if prefix.is_some() && matches!(source.canonical, Canonical::Loose { .. }) {
            return Err("use a full filename to name a file copy".to_string());
        }
        if name.is_some() && matches!(source.canonical, Canonical::Sequence(_)) {
            return Err("sequence copies are named by prefix, not filename".to_string());
        }

        let canonical = source.canonical.clone();
        let copy_id = self.insert_with_origin(canonical, Origin::CopyOf(id));

        let mut stage = |intent: IntentOp| -> Result<(), String> {
            self.stage_intent(copy_id, intent)
        };
        let result = (|| {
            if let Some(prefix) = prefix {
                stage(IntentOp::Rename {
                    to: NameComponents {
                        prefix: Some(prefix),
                        delimiter: None,
                        suffix: None,
                        extension: None,
                        padding: None,
                    },
                })?;
            }
            if let Some(name) = name {
                stage(IntentOp::RenameFile { name })?;
            }
            if let Some(dest) = dest {
                stage(IntentOp::Move { dest })?;
            }
            Ok(())
        })();

        if let Err(e) = result {
            self.entities.remove(&copy_id);
            return Err(e);
        }
        Ok(copy_id)
    }

    /// Stages an intent on an entity, validating eagerly: the fold runs
    /// immediately so an invalid edit (e.g. an offset that would underflow
    /// frame 0) is rejected at stage time, not at commit.
    pub fn stage_intent(&mut self, id: EntityId, intent: IntentOp) -> Result<(), String> {
        let entity = self
            .entities
            .get_mut(&id)
            .ok_or_else(|| format!("no entity with id {id}"))?;

        if matches!(entity.intents.last(), Some(IntentOp::Delete)) {
            return Err("entity is staged for deletion; revert the delete first".to_string());
        }
        if entity.is_copy() && matches!(intent, IntentOp::Delete) {
            return Err("discard the copy instead of deleting it".to_string());
        }
        if matches!(entity.canonical, Canonical::Loose { .. })
            && matches!(
                intent,
                IntentOp::Rename { .. }
                    | IntentOp::Offset { .. }
                    | IntentOp::Repad { .. }
                    | IntentOp::Trim { .. }
            )
        {
            return Err("that operation only applies to sequences".to_string());
        }
        // Reject the degenerate trims here so they surface as a message rather
        // than as an EmptySequence fold error or a no-op tray entry.
        if let IntentOp::Trim { first, last } = &intent {
            if first > last {
                return Err(format!(
                    "empty range: first frame {first} is after last frame {last}"
                ));
            }
            let projection = entity.project().map_err(|e| e.to_string())?;
            if let Projected::Sequence(seq) = &projection.projected {
                let kept = seq
                    .items()
                    .iter()
                    .filter(|i| (*first..=*last).contains(&i.frame_number()))
                    .count();
                if kept == 0 {
                    return Err(
                        "that range keeps no frames; delete the sequence instead".to_string()
                    );
                }
                if kept == seq.len() {
                    return Err("that range keeps every frame; nothing to trim".to_string());
                }
            }
        }
        if matches!(entity.canonical, Canonical::Sequence(_))
            && matches!(intent, IntentOp::RenameFile { .. })
        {
            return Err("sequences are renamed by components, not filename".to_string());
        }
        if let IntentOp::RenameFile { name } = &intent {
            if name.is_empty()
                || name == "."
                || name == ".."
                || name.contains('/')
                || name.contains('\\')
            {
                return Err(format!("invalid filename: {name:?}"));
            }
        }
        if matches!(
            entity.canonical,
            Canonical::Loose {
                kind: EntryKind::Symlink | EntryKind::Other,
                ..
            }
        ) {
            return Err("symlinks and special files cannot be edited".to_string());
        }
        // Recursive directory delete is app-side work (outside the
        // transactional plan) and isn't wired up yet — reject at stage time
        // rather than failing at commit.
        if matches!(intent, IntentOp::Delete)
            && matches!(
                entity.canonical,
                Canonical::Loose {
                    kind: EntryKind::Directory,
                    ..
                }
            )
        {
            return Err("directory delete is not supported yet".to_string());
        }

        // Validate (and canonicalize) a Move destination at stage time so an
        // impossible move is rejected now, not at commit. Canonicalizing also
        // keeps the stored path matching how directories are scanned, so the
        // moved entity reconciles to the same identity afterwards.
        let intent = if let IntentOp::Move { dest } = intent {
            let canon = dunce::canonicalize(&dest)
                .map_err(|_| format!("destination does not exist: {}", dest.display()))?;
            if !canon.is_dir() {
                return Err(format!("destination is not a directory: {}", canon.display()));
            }
            if entity.canonical_dir().as_deref() == Some(canon.as_path()) {
                return Err("already in that directory".to_string());
            }
            if let Canonical::Loose {
                path,
                kind: EntryKind::Directory,
                ..
            } = &entity.canonical
            {
                if canon == *path || canon.starts_with(path) {
                    return Err("cannot move a directory into itself".to_string());
                }
            }
            // A real move is fs::rename, which can't cross filesystems. Reject
            // a cross-volume move up front rather than failing at commit. (A
            // *copy*'s Move is committed as fs::copy, which is fine across
            // volumes, so copies are exempt.)
            #[cfg(unix)]
            if !entity.is_copy() {
                use std::os::unix::fs::MetadataExt;
                if let Some(src_dir) = entity.canonical_dir() {
                    if let (Ok(s), Ok(d)) =
                        (std::fs::metadata(&src_dir), std::fs::metadata(&canon))
                    {
                        if s.dev() != d.dev() {
                            return Err(
                                "cross-volume move isn't supported yet (different disk); \
                                 use Copy to… instead"
                                    .to_string(),
                            );
                        }
                    }
                }
            }
            IntentOp::Move { dest: canon }
        } else {
            intent
        };

        entity.intents.push(intent);
        if let Err(e) = entity.project() {
            entity.intents.pop();
            return Err(e.to_string());
        }
        Ok(())
    }

    /// Removes the staged intent at `index` on `id` (per-item revert).
    ///
    /// Reverting the last remaining intent of an uncommitted copy cancels the
    /// copy itself (a copy with no differentiating transform is degenerate);
    /// the entity is removed and `Ok(true)` is returned.
    pub fn revert_intent(&mut self, id: EntityId, index: usize) -> Result<bool, String> {
        let entity = self
            .entities
            .get_mut(&id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        if index >= entity.intents.len() {
            return Err(format!("no staged intent at index {index}"));
        }
        if entity.is_copy() && entity.intents.len() == 1 {
            self.entities.remove(&id);
            return Ok(true);
        }
        let removed = entity.intents.remove(index);
        // The remaining log must still fold cleanly without the removed step.
        if let Err(e) = entity.project() {
            entity.intents.insert(index, removed);
            return Err(format!("cannot revert that step: {e}"));
        }
        Ok(false)
    }

    /// Drops every staged intent on `id`. Discarding an uncommitted copy
    /// removes the entity entirely (returns `Ok(true)`) — there is nothing on
    /// disk for it to revert to.
    pub fn clear_intents(&mut self, id: EntityId) -> Result<bool, String> {
        let entity = self
            .entities
            .get_mut(&id)
            .ok_or_else(|| format!("no entity with id {id}"))?;
        if entity.is_copy() {
            self.entities.remove(&id);
            return Ok(true);
        }
        entity.intents.clear();
        Ok(false)
    }

    /// Stage-time conflict detection over the whole projected world: two
    /// entities projecting to the same path (e.g. two sequences renamed to
    /// one name, or a rename onto an unchanged neighbour) are both flagged
    /// with the contested paths. Pure — no disk access; the registry's
    /// projected state is the reference world. The commit pre-flight is the
    /// disk-reconciled backstop; this exists for live inline badges.
    pub fn projected_conflicts(&self) -> HashMap<EntityId, Vec<PathBuf>> {
        let mut by_path: HashMap<PathBuf, Vec<EntityId>> = HashMap::new();
        for (id, entity) in &self.entities {
            let Ok(paths) = entity.projected_paths() else {
                continue;
            };
            for path in paths {
                by_path.entry(path).or_default().push(*id);
            }
        }

        let mut conflicts: HashMap<EntityId, Vec<PathBuf>> = HashMap::new();
        for (path, ids) in by_path {
            if ids.len() > 1 {
                for id in ids {
                    conflicts.entry(id).or_default().push(path.clone());
                }
            }
        }
        for paths in conflicts.values_mut() {
            paths.sort();
        }
        conflicts
    }

    /// After a fully successful commit of `ids`: each entity's canonical
    /// snapshot becomes what its projection was (the disk now matches it),
    /// staged intents are cleared, and deleted entities leave the registry.
    /// Identity is carried across the commit — this is the "self-induced
    /// commit" reconciliation, deterministic because we made the changes.
    ///
    /// When a directory entity moves, any other registered entity living under
    /// its old path now points at a path that no longer exists. Rewriting a
    /// sequence's internal item paths isn't expressible through sequitur's API,
    /// so instead those subtree entities are dropped (their staged intents
    /// discarded) — a later scan of the new location re-mints them. This trades
    /// identity for safety: it guarantees no committed operation ever targets a
    /// stale path.
    pub fn finalize_commit(&mut self, ids: &[EntityId]) {
        // Record directory moves before mutating canonical state.
        let mut moved_dirs: Vec<PathBuf> = Vec::new();
        for &id in ids {
            let Some(entity) = self.entities.get(&id) else {
                continue;
            };
            if let Canonical::Loose {
                path,
                kind: EntryKind::Directory,
                ..
            } = &entity.canonical
            {
                if let Ok(proj) = entity.project() {
                    if let Projected::Loose(new_path) = &proj.projected {
                        if !proj.deleted && new_path != path {
                            moved_dirs.push(path.clone());
                        }
                    }
                }
            }
        }

        for &id in ids {
            let Some(entity) = self.entities.get_mut(&id) else {
                continue;
            };
            let Ok(projection) = entity.project() else {
                continue;
            };
            if projection.deleted {
                self.entities.remove(&id);
                continue;
            }
            match projection.projected {
                Projected::Sequence(seq) => {
                    entity.canonical = Canonical::Sequence(seq);
                }
                Projected::Loose(new_path) => {
                    if let Canonical::Loose { path, .. } = &mut entity.canonical {
                        *path = new_path;
                    }
                }
            }
            entity.intents.clear();
            // A committed copy now exists on disk: it is an ordinary entity.
            entity.origin = Origin::Disk;
        }

        // Drop entities stranded under a directory that just moved (the moved
        // directory itself now sits at its new path, so it survives).
        if !moved_dirs.is_empty() {
            self.entities.retain(|_, e| match e.canonical_dir() {
                Some(dir) => !moved_dirs.iter().any(|old| dir.starts_with(old)),
                None => true,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("seq_manip_{tag}_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn scan(dir: &Path) -> (Vec<FileSequence>, Vec<(PathBuf, EntryKind, bool)>) {
        let listing = FileSequence::parse_directory(dir, 2).unwrap();
        let entries = listing
            .entries
            .iter()
            .map(|e| (e.path.clone(), e.kind, e.hidden))
            .collect();
        (listing.sequences, entries)
    }

    #[test]
    fn fold_projects_rename_then_offset_without_io() {
        let dir = temp_dir("fold");
        for n in 1..=3 {
            fs::write(dir.join(format!("render_00{n}.exr")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(ids.len(), 1);
        let id = ids[0];

        reg.stage_intent(
            id,
            IntentOp::Rename {
                to: NameComponents {
                    prefix: Some("comp".into()),
                    delimiter: None,
                    suffix: None,
                    extension: None,
                    padding: None,
                },
            },
        )
        .unwrap();
        reg.stage_intent(
            id,
            IntentOp::Offset {
                by: 100,
                padding: None,
            },
        )
        .unwrap();

        let entity = reg.get(id).unwrap();
        let projection = entity.project().unwrap();
        let Projected::Sequence(projected) = &projection.projected else {
            panic!("expected sequence projection");
        };
        assert_eq!(projected.prefix().unwrap(), "comp");
        assert_eq!(projected.first_frame(), 101);
        assert_eq!(entity.lifecycle(&projection), Lifecycle::Modified);

        // Pure: disk is untouched.
        assert!(dir.join("render_001.exr").exists());
        assert!(!dir.join("comp_101.exr").exists());

        fs::remove_dir_all(&dir).unwrap();
    }

    /// Rename can change suffix and extension (now exposed in the UI): the
    /// projection must reflect both, leaving prefix/frames untouched.
    #[test]
    fn rename_projects_suffix_and_extension() {
        let dir = temp_dir("sfxext");
        for n in 1..=3 {
            fs::write(dir.join(format!("plate.000{n}.JPG")), b"x").unwrap();
        }
        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(ids.len(), 1);
        let id = ids[0];

        reg.stage_intent(
            id,
            IntentOp::Rename {
                to: NameComponents {
                    prefix: None,
                    delimiter: None,
                    suffix: Some("_left".into()),
                    extension: Some("jpg".into()),
                    padding: None,
                },
            },
        )
        .unwrap();

        let entity = reg.get(id).unwrap();
        let projection = entity.project().unwrap();
        let Projected::Sequence(p) = &projection.projected else {
            panic!("expected sequence projection");
        };
        assert_eq!(p.prefix().unwrap(), "plate");
        assert_eq!(p.suffix().unwrap(), Some("_left"));
        assert_eq!(p.extension().unwrap(), "jpg");
        assert_eq!(p.first_frame(), 1);
        assert_eq!(p.items()[0].filename(), "plate.0001_left.jpg");
        // Disk untouched.
        assert!(dir.join("plate.0001.JPG").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A staged delete must SURVIVE a re-scan and still be listed (as
    /// Deleted) — the UI re-scans after every stage, and the row is supposed
    /// to stay visible in red until commit, not vanish.
    #[test]
    fn staged_delete_survives_rescan_for_sequence_and_file() {
        let dir = temp_dir("stagedel");
        for n in 1..=3 {
            fs::write(dir.join(format!("render_00{n}.exr")), b"x").unwrap();
        }
        fs::write(dir.join("notes.txt"), b"x").unwrap();

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(ids.len(), 2, "one sequence + one loose file");
        for id in &ids {
            reg.stage_intent(*id, IntentOp::Delete).unwrap();
        }

        // Re-scan (nothing on disk changed — the delete is only staged).
        let (seqs, entries) = scan(&dir);
        let outcome = reg.absorb_scan(&dir, seqs, entries);
        assert!(outcome.voided.is_empty(), "nothing should be voided: {:?}", outcome.voided);
        assert_eq!(outcome.ids.len(), 2, "both entities must still be listed");
        for id in &outcome.ids {
            let e = reg.get(*id).unwrap();
            let p = e.project().unwrap();
            assert_eq!(e.lifecycle(&p), Lifecycle::Deleted, "{}", e.canonical_name());
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn invalid_offset_is_rejected_at_stage_time() {
        let dir = temp_dir("underflow");
        for n in 1..=3 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let id = reg.absorb_scan(&dir, seqs, entries).ids[0];

        let err = reg
            .stage_intent(
                id,
                IntentOp::Offset {
                    by: -10,
                    padding: None,
                },
            )
            .unwrap_err();
        assert!(err.contains("egative") || err.contains("frame"), "{err}");
        // The rejected intent must not remain staged.
        assert!(reg.get(id).unwrap().intents.is_empty());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn delete_locks_and_move_departs() {
        let dir = temp_dir("lifecycle");
        for n in 1..=2 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
        }
        fs::write(dir.join("notes.txt"), b"x").unwrap();

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        let (seq_id, file_id) = (ids[0], ids[1]);

        // Staged delete locks the entity.
        reg.stage_intent(seq_id, IntentOp::Delete).unwrap();
        let err = reg
            .stage_intent(
                seq_id,
                IntentOp::Repad { width: 5 },
            )
            .unwrap_err();
        assert!(err.contains("deletion"), "{err}");
        let entity = reg.get(seq_id).unwrap();
        let projection = entity.project().unwrap();
        assert_eq!(entity.lifecycle(&projection), Lifecycle::Deleted);

        // A move out of the dir marks the entity departing.
        let elsewhere = temp_dir("elsewhere");
        reg.stage_intent(
            file_id,
            IntentOp::Move {
                dest: elsewhere.clone(),
            },
        )
        .unwrap();
        let entity = reg.get(file_id).unwrap();
        let projection = entity.project().unwrap();
        assert_eq!(entity.lifecycle(&projection), Lifecycle::Departing);
        let Projected::Loose(path) = &projection.projected else {
            panic!("expected loose projection");
        };
        // The dest is canonicalized at stage time (resolves /tmp -> /private
        // on macOS; strips `\\?\` on Windows), so compare against the same
        // canonical form.
        let canon_elsewhere = dunce::canonicalize(&elsewhere).unwrap();
        assert_eq!(path, &canon_elsewhere.join("notes.txt"));

        fs::remove_dir_all(&dir).unwrap();
        fs::remove_dir_all(&elsewhere).unwrap();
    }

    #[test]
    fn commit_plan_is_minimal_diff_not_concatenation() {
        let dir = temp_dir("mindiff");
        for n in 1..=3 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let id = reg.absorb_scan(&dir, seqs, entries).ids[0];

        // Two stacked transforms still emit ONE rename per frame (3 ops),
        // not one per intent per frame (6).
        reg.stage_intent(
            id,
            IntentOp::Rename {
                to: NameComponents {
                    prefix: Some("b".into()),
                    delimiter: None,
                    suffix: None,
                    extension: None,
                    padding: None,
                },
            },
        )
        .unwrap();
        reg.stage_intent(
            id,
            IntentOp::Offset {
                by: 10,
                padding: None,
            },
        )
        .unwrap();
        let plan = reg.get(id).unwrap().commit_plan().unwrap();
        assert_eq!(plan.len(), 3);

        // Cancelling edits diff to nothing.
        reg.clear_intents(id).unwrap();
        reg.stage_intent(
            id,
            IntentOp::Offset {
                by: 5,
                padding: None,
            },
        )
        .unwrap();
        reg.stage_intent(
            id,
            IntentOp::Offset {
                by: -5,
                padding: None,
            },
        )
        .unwrap();
        let plan = reg.get(id).unwrap().commit_plan().unwrap();
        assert!(plan.is_empty(), "A→B→A must diff to nothing");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage_commit_finalize_round_trip() {
        let dir = temp_dir("roundtrip");
        for n in 1..=3 {
            fs::write(dir.join(format!("render_00{n}.exr")), vec![n]).unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let id = reg.absorb_scan(&dir, seqs, entries).ids[0];

        reg.stage_intent(
            id,
            IntentOp::Rename {
                to: NameComponents {
                    prefix: Some("comp".into()),
                    delimiter: None,
                    suffix: None,
                    extension: None,
                    padding: None,
                },
            },
        )
        .unwrap();

        let plan = reg.get(id).unwrap().commit_plan().unwrap();
        let result = plan.commit(false).unwrap();
        assert!(result.success());
        reg.finalize_commit(&[id]);

        // Disk reflects the rename; the entity's canonical is the new state
        // with identity preserved and the intent log drained.
        for n in 1..=3 {
            assert!(dir.join(format!("comp_00{n}.exr")).exists());
            assert!(!dir.join(format!("render_00{n}.exr")).exists());
        }
        let entity = reg.get(id).unwrap();
        assert!(entity.intents.is_empty());
        assert_eq!(entity.canonical_name(), "comp_###.exr");

        // A re-scan reconciles to the same id.
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(ids, vec![id]);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn projected_conflicts_flag_colliding_entities() {
        let dir = temp_dir("conflicts");
        for n in 1..=2 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
            fs::write(dir.join(format!("b_00{n}.exr")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        let (a, b) = (ids[0], ids[1]);
        assert!(reg.projected_conflicts().is_empty());

        // Rename a → b: collides with the UNCHANGED b sequence; both flagged.
        reg.stage_intent(
            a,
            IntentOp::Rename {
                to: NameComponents {
                    prefix: Some("b".into()),
                    delimiter: None,
                    suffix: None,
                    extension: None,
                    padding: None,
                },
            },
        )
        .unwrap();
        let conflicts = reg.projected_conflicts();
        assert_eq!(conflicts.get(&a).map(Vec::len), Some(2));
        assert_eq!(conflicts.get(&b).map(Vec::len), Some(2));

        // Deleting b vacates its paths: the rename no longer conflicts.
        reg.stage_intent(b, IntentOp::Delete).unwrap();
        assert!(reg.projected_conflicts().is_empty());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn anomalous_padding_splits_keep_distinct_identity() {
        // sequitur splits these into two sequences sharing every grouping
        // component except padding: 01/02 (pad 2) and 001/002/003 (pad 3).
        // The frame numbers overlap, so they genuinely cannot be one sequence.
        let dir = temp_dir("padsplit");
        for n in ["01", "02"] {
            fs::write(dir.join(format!("plate_{n}.dpx")), b"x").unwrap();
        }
        for n in ["001", "002", "003"] {
            fs::write(dir.join(format!("plate_{n}.dpx")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        assert_eq!(seqs.len(), 2, "expected a padding split");
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], ids[1], "split fragments must get distinct ids");

        // Stage an edit on the first, then re-scan: identity and intents must
        // survive for BOTH fragments (the bug collapsed them to one id).
        reg.stage_intent(ids[0], IntentOp::Repad { width: 5 }).unwrap();
        let (seqs, entries) = scan(&dir);
        let rescanned = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(rescanned.len(), 2);
        assert!(rescanned.contains(&ids[0]));
        assert!(rescanned.contains(&ids[1]));
        assert_eq!(reg.get(ids[0]).unwrap().intents.len(), 1);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn move_validation_rejects_bad_destination() {
        let dir = temp_dir("moveval");
        for n in 1..=2 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let id = reg.absorb_scan(&dir, seqs, entries).ids[0];

        // Nonexistent destination is rejected at stage time.
        let err = reg
            .stage_intent(
                id,
                IntentOp::Move {
                    dest: dir.join("does_not_exist"),
                },
            )
            .unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
        assert!(reg.get(id).unwrap().intents.is_empty());

        // A real directory is accepted, and the stored dest is canonicalized.
        let sub = dir.join("sub");
        fs::create_dir(&sub).unwrap();
        reg.stage_intent(id, IntentOp::Move { dest: sub.clone() })
            .unwrap();
        let entity = reg.get(id).unwrap();
        assert_eq!(entity.intents.len(), 1);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn copy_round_trip_creates_files_and_keeps_source() {
        let dir = temp_dir("copyrt");
        for n in 1..=3 {
            fs::write(dir.join(format!("a_00{n}.exr")), vec![n]).unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let src = reg.absorb_scan(&dir, seqs, entries).ids[0];

        // Duplicate in place under a new prefix.
        let copy = reg.stage_copy(src, None, Some("b".into()), None).unwrap();
        assert_ne!(copy, src);
        let entity = reg.get(copy).unwrap();
        assert!(entity.is_copy());
        let projection = entity.project().unwrap();
        assert_eq!(entity.lifecycle(&projection), Lifecycle::New);
        // Visible in the directory it will land in.
        assert!(reg.copies_in(&dir).contains(&copy));

        // Plan is Copy ops; committing creates the files, source untouched.
        let plan = reg.get(copy).unwrap().commit_plan().unwrap();
        assert_eq!(plan.len(), 3);
        assert!(plan.commit(false).unwrap().success());
        reg.finalize_commit(&[copy]);

        for n in 1..=3 {
            assert_eq!(fs::read(dir.join(format!("b_00{n}.exr"))).unwrap(), vec![n]);
            assert!(dir.join(format!("a_00{n}.exr")).exists());
        }
        // Now an ordinary entity; a re-scan reconciles both sequences.
        assert!(!reg.get(copy).unwrap().is_copy());
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert!(ids.contains(&src) && ids.contains(&copy));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn copy_then_source_rename_commits_in_one_plan() {
        // The copy reads paths the source is simultaneously vacating: the
        // scheduler must run the copies before the renames.
        let dir = temp_dir("copymove");
        for n in 1..=2 {
            fs::write(dir.join(format!("a_00{n}.exr")), vec![n]).unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let src = reg.absorb_scan(&dir, seqs, entries).ids[0];
        let copy = reg.stage_copy(src, None, Some("backup".into()), None).unwrap();
        reg.stage_intent(
            src,
            IntentOp::Rename {
                to: NameComponents {
                    prefix: Some("renamed".into()),
                    delimiter: None,
                    suffix: None,
                    extension: None,
                    padding: None,
                },
            },
        )
        .unwrap();

        let mut plan = sequitur::OperationPlan::new();
        for (_, entity) in reg.pending() {
            plan.extend(entity.commit_plan().unwrap());
        }
        assert!(plan.commit(false).unwrap().success());
        reg.finalize_commit(&[src, copy]);

        for n in 1..=2 {
            assert!(dir.join(format!("backup_00{n}.exr")).exists());
            assert!(dir.join(format!("renamed_00{n}.exr")).exists());
            assert!(!dir.join(format!("a_00{n}.exr")).exists());
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn degenerate_copy_conflicts_and_discard_removes_it() {
        let dir = temp_dir("copydegen");
        for n in 1..=2 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let src = reg.absorb_scan(&dir, seqs, entries).ids[0];

        // No name and no destination → rejected outright.
        assert!(reg.stage_copy(src, None, None, None).is_err());

        // Reverting a copy's only transform cancels the copy.
        let copy = reg.stage_copy(src, None, Some("b".into()), None).unwrap();
        assert!(reg.revert_intent(copy, 0).unwrap());
        assert!(reg.get(copy).is_none());

        // Discard removes it too.
        let copy = reg.stage_copy(src, None, Some("b".into()), None).unwrap();
        assert!(reg.clear_intents(copy).unwrap());
        assert!(reg.get(copy).is_none());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn loose_rename_round_trip() {
        let dir = temp_dir("looserename");
        fs::write(dir.join("a_001.exr"), b"x").unwrap();
        fs::write(dir.join("a_002.exr"), b"x").unwrap();
        fs::write(dir.join("notes.txt"), b"n").unwrap();

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        let file_id = *ids.last().unwrap();

        // Bad names rejected; sequences reject filename renames.
        assert!(reg
            .stage_intent(file_id, IntentOp::RenameFile { name: "a/b".into() })
            .is_err());
        assert!(reg
            .stage_intent(ids[0], IntentOp::RenameFile { name: "x.txt".into() })
            .is_err());

        reg.stage_intent(
            file_id,
            IntentOp::RenameFile {
                name: "readme.txt".into(),
            },
        )
        .unwrap();
        let plan = reg.get(file_id).unwrap().commit_plan().unwrap();
        assert_eq!(plan.len(), 1);
        assert!(plan.commit(false).unwrap().success());
        reg.finalize_commit(&[file_id]);

        assert!(dir.join("readme.txt").exists());
        assert!(!dir.join("notes.txt").exists());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn external_change_reports_voided_intents() {
        let dir = temp_dir("voided");
        fs::write(dir.join("a_001.exr"), b"x").unwrap();
        fs::write(dir.join("a_002.exr"), b"x").unwrap();

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let id = reg.absorb_scan(&dir, seqs, entries).ids[0];
        reg.stage_intent(id, IntentOp::Delete).unwrap();

        // The files vanish externally; the next scan voids the staged edits
        // and says so.
        fs::remove_file(dir.join("a_001.exr")).unwrap();
        fs::remove_file(dir.join("a_002.exr")).unwrap();
        let (seqs, entries) = scan(&dir);
        let outcome = reg.absorb_scan(&dir, seqs, entries);
        assert!(outcome.ids.is_empty());
        assert_eq!(outcome.voided.len(), 1);
        assert!(outcome.voided[0].contains("a_"), "{:?}", outcome.voided);
        assert!(reg.get(id).is_none());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn loose_duplicate_via_copy_name() {
        let dir = temp_dir("loosedup");
        fs::write(dir.join("notes.txt"), b"n").unwrap();

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        let file_id = *ids.last().unwrap();

        let copy = reg
            .stage_copy(file_id, None, None, Some("notes_copy.txt".into()))
            .unwrap();
        let plan = reg.get(copy).unwrap().commit_plan().unwrap();
        assert!(plan.commit(false).unwrap().success());
        reg.finalize_commit(&[copy]);

        assert_eq!(fs::read(dir.join("notes_copy.txt")).unwrap(), b"n");
        assert_eq!(fs::read(dir.join("notes.txt")).unwrap(), b"n");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rescan_preserves_identity_and_intents() {
        let dir = temp_dir("rescan");
        for n in 1..=3 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
        }

        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let id = reg.absorb_scan(&dir, seqs, entries).ids[0];
        reg.stage_intent(
            id,
            IntentOp::Repad { width: 5 },
        )
        .unwrap();

        // External change: a frame appears. Re-scan.
        fs::write(dir.join("a_004.exr"), b"x").unwrap();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;

        // Same identity, intents survive, snapshot refreshed to 4 frames.
        assert_eq!(ids, vec![id]);
        let entity = reg.get(id).unwrap();
        assert_eq!(entity.intents.len(), 1);
        let Canonical::Sequence(seq) = &entity.canonical else {
            panic!("expected sequence");
        };
        assert_eq!(seq.actual_frame_count(), 4);

        // A genuinely new group mints a new id; the old one persists.
        fs::write(dir.join("b_001.exr"), b"x").unwrap();
        fs::write(dir.join("b_002.exr"), b"x").unwrap();
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id));

        // A vanished group is dropped from the registry.
        for n in 1..=4 {
            fs::remove_file(dir.join(format!("a_00{n}.exr"))).unwrap();
        }
        let (seqs, entries) = scan(&dir);
        let ids = reg.absorb_scan(&dir, seqs, entries).ids;
        assert_eq!(ids.len(), 1);
        assert!(!ids.contains(&id));
        assert!(reg.get(id).is_none());

        fs::remove_dir_all(&dir).unwrap();
    }

    /// Stages a 6-frame sequence `a_001..a_006` and returns its registry entry.
    fn six_frame_registry(tag: &str) -> (PathBuf, Registry, EntityId) {
        let dir = temp_dir(tag);
        for n in 1..=6 {
            fs::write(dir.join(format!("a_00{n}.exr")), b"x").unwrap();
        }
        let mut reg = Registry::new();
        let (seqs, entries) = scan(&dir);
        let id = reg.absorb_scan(&dir, seqs, entries).ids[0];
        (dir, reg, id)
    }

    #[test]
    fn trim_deletes_dropped_frames_and_leaves_survivors_untouched() {
        let (dir, mut reg, id) = six_frame_registry("trim");

        reg.stage_intent(id, IntentOp::Trim { first: 4, last: 6 })
            .unwrap();

        let entity = reg.get(id).unwrap();
        let projection = entity.project().unwrap();
        let Projected::Sequence(projected) = &projection.projected else {
            panic!("expected sequence projection");
        };
        // Survivors keep their own numbers — trim never renumbers.
        assert_eq!(projected.first_frame(), 4);
        assert_eq!(projected.last_frame(), 6);
        assert_eq!(projected.len(), 3);
        assert_eq!(entity.lifecycle(&projection), Lifecycle::Modified);

        // The head is deleted and nothing is renamed.
        let plan = entity.commit_plan().unwrap();
        let deleted: Vec<&PathBuf> = plan
            .operations()
            .iter()
            .filter_map(|op| match op {
                FileOperation::Delete { source } => Some(source),
                _ => None,
            })
            .collect();
        assert_eq!(deleted.len(), 3);
        for n in 1..=3 {
            assert!(deleted.contains(&&dir.join(format!("a_00{n}.exr"))));
        }
        assert_eq!(plan.operations().len(), 3, "survivors must not be renamed");

        fs::remove_dir_all(&dir).unwrap();
    }

    /// The pairing regression: trim changes the frame count, so a following
    /// offset must still map each SURVIVING canonical frame to its own new
    /// name. Pairing canonical-to-projected positionally without dropping the
    /// trimmed frames first would rename the wrong files.
    #[test]
    fn trim_then_offset_renames_survivors_from_the_right_sources() {
        let (dir, mut reg, id) = six_frame_registry("trimoffset");

        reg.stage_intent(id, IntentOp::Trim { first: 4, last: 6 })
            .unwrap();
        reg.stage_intent(
            id,
            IntentOp::Offset {
                by: 100,
                padding: None,
            },
        )
        .unwrap();

        let plan = reg.get(id).unwrap().commit_plan().unwrap();
        let renames: Vec<(&PathBuf, &PathBuf)> = plan
            .operations()
            .iter()
            .filter_map(|op| match op {
                FileOperation::Rename {
                    source,
                    destination,
                } => Some((source, destination)),
                _ => None,
            })
            .collect();
        // 4->104, 5->105, 6->106; frames 1-3 are deleted, not renamed.
        assert_eq!(renames.len(), 3);
        for (n, expected) in [(4, 104), (5, 105), (6, 106)] {
            let src = dir.join(format!("a_00{n}.exr"));
            let dst = dir.join(format!("a_{expected}.exr"));
            assert!(
                renames.contains(&(&src, &dst)),
                "expected {} -> {}, got {renames:?}",
                src.display(),
                dst.display()
            );
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    /// A copy trims by simply not copying the dropped frames: the originals
    /// belong to the source entity and must survive.
    #[test]
    fn trimming_a_copy_deletes_nothing() {
        let (dir, mut reg, id) = six_frame_registry("trimcopy");

        let copy = reg.stage_copy(id, None, Some("b".into()), None).unwrap();
        reg.stage_intent(copy, IntentOp::Trim { first: 4, last: 6 })
            .unwrap();

        let plan = reg.get(copy).unwrap().commit_plan().unwrap();
        assert!(
            !plan
                .operations()
                .iter()
                .any(|op| matches!(op, FileOperation::Delete { .. })),
            "a trimmed copy must not delete the source frames: {:?}",
            plan.operations()
        );
        // Only the kept frames are copied.
        assert_eq!(plan.operations().len(), 3);

        fs::remove_dir_all(&dir).unwrap();
    }

    /// The dangerous interleaving, executed for real: trim drops frames 1-3,
    /// then an offset moves the survivors onto exactly those paths. The deletes
    /// have to free the head before the renames land on it — sequitur's
    /// scheduler orders that, but only if the plan carries both halves.
    /// (This one really deletes, so it routes through the OS trash.)
    #[test]
    fn committing_trim_then_offset_lands_survivors_on_the_freed_head() {
        let (dir, mut reg, id) = six_frame_registry("trimcommit");
        // Distinct contents, so we can prove which frames survived.
        for n in 1..=6 {
            fs::write(dir.join(format!("a_00{n}.exr")), format!("frame{n}")).unwrap();
        }

        reg.stage_intent(id, IntentOp::Trim { first: 4, last: 6 })
            .unwrap();
        reg.stage_intent(
            id,
            IntentOp::Offset {
                by: -3,
                padding: None,
            },
        )
        .unwrap();

        let plan = reg.get(id).unwrap().commit_plan().unwrap();
        let result = plan.commit(false).unwrap();
        assert!(result.success(), "commit failed: {result:?}");

        // Frames 4-6 now occupy 1-3, carrying their own bytes.
        for (now, was) in [(1, 4), (2, 5), (3, 6)] {
            let path = dir.join(format!("a_00{now}.exr"));
            assert_eq!(
                fs::read_to_string(&path).unwrap(),
                format!("frame{was}"),
                "a_00{now}.exr should hold what was frame {was}"
            );
        }
        // And nothing is left above them.
        for n in 4..=6 {
            assert!(!dir.join(format!("a_00{n}.exr")).exists());
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn degenerate_trims_are_rejected_at_stage_time() {
        let (dir, mut reg, id) = six_frame_registry("trimreject");

        assert!(reg
            .stage_intent(id, IntentOp::Trim { first: 6, last: 4 })
            .unwrap_err()
            .contains("empty range"));
        assert!(reg
            .stage_intent(id, IntentOp::Trim { first: 90, last: 99 })
            .unwrap_err()
            .contains("keeps no frames"));
        assert!(reg
            .stage_intent(id, IntentOp::Trim { first: 1, last: 6 })
            .unwrap_err()
            .contains("nothing to trim"));
        // None of them left a staged intent behind.
        assert!(reg.get(id).unwrap().intents.is_empty());

        fs::remove_dir_all(&dir).unwrap();
    }
}
