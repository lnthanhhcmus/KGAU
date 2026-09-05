"""Checkpoint key remapping for legacy SimKGC ``CustomBertModel`` checkpoints."""

from collections import OrderedDict


_LEGACY_PREFIX_MAP = {
	'hr_bert.': 'rel_embedder.encoder.',
	'tail_bert.': 'ent_embedder.encoder.',
	'log_inv_t': 'contrastive_state.log_inv_t',
	'pre_batch_vectors': 'contrastive_state.pre_batch_vectors',
}


def _is_legacy_simkgc_state(state_dict: dict) -> bool:
	keys = list(state_dict.keys())
	if any(key.startswith('hr_bert.') or key.startswith('tail_bert.') for key in keys):
		return True
	if any(key == 'log_inv_t' for key in keys) and not any(key.startswith('contrastive_state.') for key in keys):
		return True
	return False


def migrate_simkgc_state_dict(state_dict: dict) -> tuple[dict, bool]:
	"""Remap legacy monolithic SimKGC keys to TextKGEModel layout."""

	if not _is_legacy_simkgc_state(state_dict):
		return state_dict, False

	migrated = OrderedDict()
	for key, value in state_dict.items():
		new_key = key
		for old_prefix, new_prefix in _LEGACY_PREFIX_MAP.items():
			if key == old_prefix or key.startswith(old_prefix):
				new_key = new_prefix + key[len(old_prefix):]
				break
		migrated[new_key] = value
	return migrated, True
