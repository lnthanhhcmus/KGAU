"""Preprocessing scripts for generic KG datasets."""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripleExample:
    """Data class representing a single triple example, including IDs, text, and descriptions for the head and tail entities, as well as the relation."""

    head_id: str
    relation_id: str
    tail_id: str
    head: str
    relation: str
    tail: str
    head_desc: str = ""
    tail_desc: str = ""
    label: Optional[str] = None


def _parse_entity_name(entity: str, task: str = "") -> str:
    """Parse an entity name from its raw string representation, applying dataset-specific rules if necessary. For WN18RR, removes the last two underscore-separated tokens from the entity string. For other datasets, returns the entity string as is (or an empty string if the input is None)."""

    if task.lower() == 'wn18rr':
        return ' '.join(entity.split('_')[:-2])
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    """Concatenate an entity name and description into a single string, ensuring that the description does not redundantly include the entity name. If the description starts with the entity name, it is removed to avoid redundancy. The resulting string is in the format 'entity: description' if there is a non-empty description, or just 'entity' if the description is empty after removing redundancy."""
    
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def _normalize_whitespace(text: Optional[str]) -> str:
    """Normalize whitespace in a string, replacing sequences of whitespace with a single space and stripping leading/trailing whitespace."""

    if text is None:
        return ""
    return " ".join(str(text).split())


def _truncate(text: Optional[str], max_tokens: int) -> str:
    """Truncate a string to a maximum number of tokens, where tokens are defined as whitespace-separated substrings. Also normalizes whitespace before truncation."""

    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""
    return " ".join(normalized.split()[:max_tokens])


def _read_lines(path: str) -> List[str]:
    """Read lines from a text file, returning a list of strings."""

    with open(path, "r", encoding="utf-8") as reader:
        return reader.readlines()


def _read_tab_mapping(path: str, *, join_rest: bool = False) -> Dict[str, str]:
    """Read a tab-separated key-value mapping from a text file. Each line should contain at least two fields, where the first field is the key and the second field (or the rest of the line if `join_rest` is True) is the value. Returns a dictionary mapping keys to values."""
    
    mapping: Dict[str, str] = {}
    for line in _read_lines(path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 2:
            raise ValueError(f"Invalid line in {path}: {line.rstrip()}")
        key = fields[0]
        value = "\t".join(fields[1:]) if join_rest else fields[1]
        mapping[key] = _normalize_whitespace(value)
    return mapping


def _load_triples(path: str) -> List[Tuple[str, str, str, str]]:
    """Load triples from a text file, where each line contains three or four tab-separated fields: head_id, relation_id, tail_id, and optionally a label. Returns a list of (head_id, relation_id, tail_id, label) tuples."""

    triples: List[Tuple[str, str, str, str]] = []
    for line in _read_lines(path):
        fields = line.strip().split("\t")
        if len(fields) not in (3, 4):
            raise ValueError(f"Expected 3 or 4 tab-separated fields in {path}: {line.strip()}")
        head_id, relation_id, tail_id = fields[:3]
        label = fields[3] if len(fields) == 4 else ""
        triples.append((head_id, relation_id, tail_id, label))
    return triples


def _normalize_fb15k237_relation(relation: str) -> str:
    """Normalize FB15k-237 relation strings by replacing certain characters and removing redundant tokens."""

    tokens = [
        token
        for token in relation.replace("./", "/").replace("_", " ").strip().split("/")
        if token
    ]
    dedup_tokens: List[str] = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    relation_tokens = dedup_tokens[::-1]
    return " ".join(
        token
        for idx, token in enumerate(relation_tokens)
        if idx == 0 or token != relation_tokens[idx - 1]
    )


def _build_id_map(values: Iterable[str]) -> Dict[str, int]:
    """Build a mapping from string values to unique integer IDs based on the order of first occurrence."""

    mapping: Dict[str, int] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def _map_examples(
    triples: Sequence[Tuple[str, str, str, str]],
    *,
    entity_text: Dict[str, str],
    relation_text: Dict[str, str],
    entity_desc: Dict[str, str],
    workers: int,
    relation_transform=None,
) -> List[TripleExample]:
    """Map raw triples to TripleExample instances, enriching with text and descriptions. Uses multithreading if workers > 1."""

    def build_example(triple: Tuple[str, str, str, str]) -> TripleExample:
        """Build a TripleExample from a raw triple, applying text lookups and transformations."""

        head_id, relation_id, tail_id, label = triple
        relation_display = relation_transform(relation_id) if relation_transform else relation_text.get(relation_id, relation_id)
        return TripleExample(
            head_id=head_id,
            relation_id=relation_id,
            tail_id=tail_id,
            head=entity_text.get(head_id, head_id),
            relation=relation_display,
            tail=entity_text.get(tail_id, tail_id),
            head_desc=entity_desc.get(head_id, ""),
            tail_desc=entity_desc.get(tail_id, ""),
            label=label or None,
        )

    if workers <= 1:
        return [build_example(triple) for triple in triples]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(build_example, triples))


def _collect_metadata(examples: Sequence[TripleExample]) -> Tuple[List[dict], List[dict]]:
    """Collect unique entities and relations from the examples, returning lists of metadata dictionaries for entities and relations."""

    entities: Dict[str, dict] = {}
    relations: Dict[str, dict] = {}

    for ex in examples:
        if ex.head_id not in entities:
            entities[ex.head_id] = {
                "entity_id": ex.head_id,
                "entity": ex.head,
                "entity_desc": ex.head_desc,
            }
        if ex.tail_id not in entities:
            entities[ex.tail_id] = {
                "entity_id": ex.tail_id,
                "entity": ex.tail,
                "entity_desc": ex.tail_desc,
            }
        if ex.relation_id not in relations:
            relations[ex.relation_id] = {
                "relation_id": ex.relation_id,
                "relation": ex.relation,
            }

    return list(entities.values()), list(relations.values())


def _save_json(path: str, payload) -> None:
    """Save a Python object as a JSON file, creating the directory if it doesn't exist."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=4)


def _load_wn18rr_metadata(data_dir: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load WN18RR metadata from the specified data directory, returning dictionaries for entity text, relation text, and entity descriptions."""

    entity_text: Dict[str, str] = {}
    entity_desc: Dict[str, str] = {}
    path = os.path.join(data_dir, "wordnet-mlj12-definitions.txt")
    for line in _read_lines(path):
        fields = line.strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"Invalid line in {path}: {line.strip()}")
        entity_id, word, desc = fields
        entity_text[entity_id] = word.replace("__", " ")
        entity_desc[entity_id] = desc
    return entity_text, {}, entity_desc


def _load_fb15k237_metadata(data_dir: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load FB15k-237 metadata from the specified data directory, returning dictionaries for entity text, relation text, and entity descriptions."""
    
    names_path = os.path.join(data_dir, "FB15k_mid2name.txt")
    desc_path = os.path.join(data_dir, "FB15k_mid2description.txt")

    entity_text = {
        entity_id: _normalize_whitespace(name).replace("_", " ")
        for entity_id, name in _read_tab_mapping(names_path).items()
    }
    entity_desc = {
        entity_id: _truncate(desc, 50)
        for entity_id, desc in _read_tab_mapping(desc_path, join_rest=True).items()
    }
    return entity_text, {}, entity_desc


def _load_wiki5m_metadata(data_dir: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load Wiki5M metadata from the specified data directory, returning dictionaries for entity text, relation text, and entity descriptions."""

    entity_text = {
        entity_id: _truncate(name, 10)
        for entity_id, name in _read_tab_mapping(os.path.join(data_dir, "wikidata5m_entity.txt")).items()
    }
    relation_text = {
        relation_id: _truncate(name, 10)
        for relation_id, name in _read_tab_mapping(os.path.join(data_dir, "wikidata5m_relation.txt")).items()
    }
    entity_desc = {
        entity_id: _truncate(text, 30)
        for entity_id, text in _read_tab_mapping(os.path.join(data_dir, "wikidata5m_text.txt"), join_rest=True).items()
    }
    return entity_text, relation_text, entity_desc


def _load_generic_metadata(args) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load generic metadata based on optional file paths provided in args, returning dictionaries for entity text, relation text, and entity descriptions."""

    entity_text: Dict[str, str] = {}
    relation_text: Dict[str, str] = {}
    entity_desc: Dict[str, str] = {}

    if args.entity_text_path and os.path.exists(args.entity_text_path):
        entity_text.update(_read_tab_mapping(args.entity_text_path, join_rest=True))
    if args.relation_text_path and os.path.exists(args.relation_text_path):
        relation_text.update(_read_tab_mapping(args.relation_text_path, join_rest=True))
    if args.entity_desc_path and os.path.exists(args.entity_desc_path):
        entity_desc.update(_read_tab_mapping(args.entity_desc_path, join_rest=True))

    return entity_text, relation_text, entity_desc


def load_entity_ids_from_definitions(def_path: str) -> Set[str]:
    """Load entity IDs from a definitions file, where each line contains at least three tab-separated fields: entity_id, entity_name, and entity_description. Returns a set of entity IDs."""

    ids: Set[str] = set()
    for line in _read_lines(def_path):
        fields = line.strip().split('\t')
        if len(fields) == 3:
            ids.add(fields[0])
    return ids


def load_entity_ids_from_split(split_path: str) -> Set[str]:
    """Load entity IDs from a split file, where each line contains three tab-separated fields: head_id, relation_id, and tail_id. Returns a set of entity IDs."""

    ids: Set[str] = set()
    for line in _read_lines(split_path):
        fields = line.strip().split('\t')
        if len(fields) == 3:
            ids.add(fields[0])
            ids.add(fields[2])
    return ids


def check_missing_entity_ids(def_path: str, split_paths: Sequence[str]) -> Dict[str, List[str]]:
    """Check for missing entity IDs in the split files compared to the definitions file. Returns a dictionary mapping each split path to a list of missing entity IDs."""

    def_ids = load_entity_ids_from_definitions(def_path)
    missing_by_split: Dict[str, List[str]] = {}
    for split_path in split_paths:
        split_ids = load_entity_ids_from_split(split_path)
        missing = sorted(split_ids - def_ids)
        missing_by_split[split_path] = missing
        if missing:
            logger.info(f"Missing in {split_path}: {set(missing)}")
        else:
            logger.info(f"All entity IDs in {split_path} are present in definitions.")
    return missing_by_split


def _resolve_output_dir(args) -> str:
    """Resolve the output directory for processed files based on the provided arguments."""

    if args.output_dir:
        return args.output_dir

    data_dir = _resolve_data_dir(args)
    return os.path.join(data_dir, "preprocessed")


def _replace_split_suffix(path: str, source_suffix: str, target_suffix: str) -> str:
    """Swap one dataset split suffix for another while preserving the directory."""

    if not path:
        return path

    directory, basename = os.path.split(path)
    if source_suffix not in basename:
        return path
    return os.path.join(directory, basename.replace(source_suffix, target_suffix))


def _resolve_data_dir(args) -> str:
    """Resolve the dataset directory, preferring an explicit --data-dir and otherwise falling back to the dataset preset."""

    if args.data_dir:
        return args.data_dir

    dataset = (args.dataset or "generic").strip().lower()
    preset_dirs = {
        "wn18rr": "WN18RR",
        "fb15k237": "FB15k237",
        "wiki5m_trans": "wiki5m_trans",
        "wiki5m_ind": "wiki5m_ind",
    }
    dataset_dir = preset_dirs.get(dataset, args.dataset or "generic")
    return os.path.join("data", dataset_dir)


def _has_entries(path: str) -> bool:
    """Return True when a directory exists and contains at least one entry."""

    if not os.path.isdir(path):
        return False
    with os.scandir(path) as entries:
        return any(True for _ in entries)


def _output_has_expected_label_schema(output_dir: str) -> bool:
    """Return True when cached preprocessing outputs match the current label schema."""

    labeled_splits = {"valid_w_label.txt.json", "test_w_label.txt.json"}
    unlabeled_splits = {"train.txt.json", "valid.txt.json", "test.txt.json"}

    for split_name in labeled_splits | unlabeled_splits:
        split_path = os.path.join(output_dir, split_name)
        if not os.path.exists(split_path):
            return False

        with open(split_path, "r", encoding="utf-8") as reader:
            try:
                payload = json.load(reader)
            except json.JSONDecodeError:
                return False

        if not isinstance(payload, list) or not payload:
            return False

        first_example = payload[0]
        if not isinstance(first_example, dict):
            return False

        has_label = "label" in first_example
        if split_name in labeled_splits and not has_label:
            return False
        if split_name in unlabeled_splits and has_label:
            return False

    return True


def _resolve_split_input(data_dir: str, explicit_path: str, fallback_names: Sequence[str]) -> str:
    """Resolve a split path from an explicit CLI argument or common files under the data directory."""

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
        candidates.append(os.path.join(data_dir, os.path.basename(explicit_path)))

    for fallback_name in fallback_names:
        candidates.append(os.path.join(data_dir, fallback_name))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return explicit_path if explicit_path and os.path.exists(explicit_path) else ""


def preprocess_dataset(args) -> None:
    """Preprocess the dataset according to the specified arguments, including loading metadata, mapping examples, and saving processed files."""

    dataset = (args.dataset or "generic").lower()
    data_dir = _resolve_data_dir(args)
    output_dir = _resolve_output_dir(args)

    if _has_entries(output_dir) and _output_has_expected_label_schema(output_dir):
        logger.info(f"Dataset has been preprocessed and saved in {output_dir}")
        return

    if dataset == "wn18rr":
        entity_text, relation_text, entity_desc = _load_wn18rr_metadata(data_dir)
        relation_transform = None
    elif dataset == "fb15k237":
        entity_text, relation_text, entity_desc = _load_fb15k237_metadata(data_dir)
        relation_transform = _normalize_fb15k237_relation
    elif dataset in {"wiki5m_trans", "wiki5m_ind"}:
        entity_text, relation_text, entity_desc = _load_wiki5m_metadata(data_dir)
        relation_transform = None
    else:
        entity_text, relation_text, entity_desc = _load_generic_metadata(args)
        relation_transform = None

    split_paths = [
        ("train", _resolve_split_input(data_dir, args.train_path, ["train.txt"])),
        ("valid", _resolve_split_input(data_dir, args.valid_path, ["valid.txt"])),
        ("test", _resolve_split_input(data_dir, args.test_path, ["test.txt"])),
        ("valid_w_label", _resolve_split_input(data_dir, getattr(args, "valid_w_label_path", ""), ["valid_w_label.txt"])),
        ("test_w_label", _resolve_split_input(data_dir, getattr(args, "test_w_label_path", ""), ["test_w_label.txt"])),
    ]
    split_paths = [(split_name, path) for split_name, path in split_paths if path]
    if not split_paths:
        raise ValueError("At least one of --train-path, --valid-path, or --test-path must be provided")

    split_examples: List[Tuple[str, str, List[TripleExample]]] = []
    all_triples: List[Tuple[str, str, str, str]] = []

    for split_name, path in split_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        triples = _load_triples(path)
        examples = _map_examples(
            triples,
            entity_text=entity_text,
            relation_text=relation_text,
            entity_desc=entity_desc,
            workers=max(1, args.workers),
            relation_transform=relation_transform,
        )
        split_examples.append((split_name, path, examples))
        all_triples.extend(triples)

    entity2id = _build_id_map([head_id for head_id, _, _, _ in all_triples] + [tail_id for _, _, tail_id, _ in all_triples])
    relation2id = _build_id_map([relation_id for _, relation_id, _, _ in all_triples])

    os.makedirs(output_dir, exist_ok=True)

    metadata_entities, metadata_relations = _collect_metadata([ex for _, _, examples in split_examples for ex in examples])

    _save_json(os.path.join(output_dir, "entity2id.json"), entity2id)
    _save_json(os.path.join(output_dir, "relation2id.json"), relation2id)
    _save_json(os.path.join(output_dir, "entities.json"), metadata_entities)
    _save_json(os.path.join(output_dir, "relations.json"), metadata_relations)

    for split_name, path, examples in split_examples:
        out_path = os.path.join(output_dir, f"{os.path.basename(path)}.json")
        payload = []
        for example in examples:
            example_dict = asdict(example)
            if example_dict.get("label") is None:
                example_dict.pop("label", None)
            payload.append(example_dict)
        _save_json(out_path, payload)
        logger.info(f"Save {len(examples)} examples to {out_path}")

    logger.info(f"Save {len(entity2id)} entities to {os.path.join(output_dir, 'entity2id.json')}")
    logger.info(f"Save {len(relation2id)} relations to {os.path.join(output_dir, 'relation2id.json')}")


def check_missing_entities_from_args(args) -> Dict[str, List[str]]:
    """Check for missing entity IDs based on the provided arguments, including definitions path and split paths. Returns a dictionary mapping each split path to a list of missing entity IDs."""

    def_path = args.definitions_path
    if not def_path and (args.dataset or '').lower() == 'wn18rr':
        base_dir = _resolve_data_dir(args)
        def_path = os.path.join(base_dir, 'wordnet-mlj12-definitions.txt')

    split_paths = list(args.missing_entity_paths or [])
    if not split_paths:
        split_paths = [p for p in [args.train_path, args.valid_path, args.test_path] if p]

    if not def_path:
        raise ValueError('Please provide --definitions-path or use the WN18RR preset')
    if not split_paths:
        raise ValueError('Please provide at least one split path to check')

    return check_missing_entity_ids(def_path, split_paths)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the preprocessing script, defining all necessary command-line arguments."""

    parser = argparse.ArgumentParser(description="Generic KG preprocessing")
    parser.add_argument(
        "--dataset",
        "--dataset",
        dest="dataset",
        default="generic",
        type=str,
        help="dataset preset: generic, wn18rr, fb15k237, wiki5m_trans, wiki5m_ind",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        type=str,
        help="directory containing dataset metadata files; defaults to data/<dataset>/ when omitted",
    )
    parser.add_argument("--train-path", default="", type=str, help="path to training triples")
    parser.add_argument("--valid-path", default="", type=str, help="path to validation triples")
    parser.add_argument("--test-path", default="", type=str, help="path to test triples")
    parser.add_argument("--valid-w-label-path", default="", type=str, help="path to validation triples for triple classification")
    parser.add_argument("--test-w-label-path", default="", type=str, help="path to test triples for triple classification")
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="directory where processed files will be written; defaults to data/<dataset>/preprocessed/ when omitted",
    )
    parser.add_argument("--workers", default=1, type=int, help="number of threads for example mapping")
    parser.add_argument("--entity-text-path", default="", type=str, help="optional entity text file for generic mode")
    parser.add_argument("--entity-desc-path", default="", type=str, help="optional entity description file for generic mode")
    parser.add_argument("--relation-text-path", default="", type=str, help="optional relation text file for generic mode")
    parser.add_argument("--definitions-path", default="", type=str, help="entity definitions file for missing-entity checks")
    parser.add_argument("--missing-entity-paths", nargs='*', default=None, help="raw split files to inspect for missing entities")
    parser.add_argument("--check-missing-entities", action='store_true', help="only run the missing entity checker")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.check_missing_entities:
        check_missing_entities_from_args(args)
        return
    preprocess_dataset(args)


if __name__ == "__main__":
    main()