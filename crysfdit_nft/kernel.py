"""Frozen chemical-substitution kernel (arm Sall).

Each generated crystal keeps its geometry and its atom count. The kernel draws up to ``max_edits`` whole-species swaps
from ``assets/element_rules.json``: a species is replaced by a compatible element of the same group and block that is
not yet in the crystal, so every swap preserves the arity and the group/block composition. The lattice is then scaled
by the ratio of mean atomic radii, clamped to [0.85, 1.15]. No energy, novelty or other oracle is used, and every input
row gives exactly one output row.

The random stream of row ``index`` is ``random.Random(proposal_seed + 1009 * index)``; the reported results use
``proposal_seed = sample_seed + 50`` and ``max_edits = 20``.
"""
import copy
import math
import random
from collections import Counter


def ordered_sum(values):
    """Left-to-right float sum (Python 3.12 changed the accumulation order of the built-in ``sum`` for floats)."""
    total = 0.0
    for value in values:
        total += value
    return total


class ChemicalKernel:
    def __init__(self, rules):
        self.meta = {int(k): v for k, v in rules['elements'].items()}
        self.compat = {int(k): v for k, v in rules['compatibility'].items()}

    def group(self, z):
        e = self.meta.get(z)
        return (e['group'], e['block']) if e else ('unsupported', z)

    def apply(self, original, index, proposal_seed, max_edits=20, arm='Sall'):
        assert 'chemical_sampling' not in (original.get('metadata') or {}), 'apply the kernel to model outputs only once'
        if original.get('generation_failed') or not all(k in original for k in ('atomic_numbers', 'coords', 'lattice')):
            row = copy.deepcopy(original)
            row['metadata'] = dict(row.get('metadata') or {},
                                   chemical_sampling={'arm': arm, 'actions': [], 'failure_pass_through': True})
            return row
        old = original['atomic_numbers']
        original_species = set(old)
        current = list(old)
        actions = []
        used = set()
        rng = random.Random(proposal_seed + 1009 * index)
        for _ in range(max_edits):
            feasible = {}
            for z in sorted(original_species - used):
                options = [x for x in self.compat.get(z, []) if x['target'] not in original_species and x['target'] not in current]
                if options:
                    feasible[z] = options
            if not feasible:
                break
            source = rng.choice(sorted(feasible))
            options = feasible[source]
            option = rng.choices(options, weights=[x['weight'] for x in options], k=1)[0]
            target = option['target']
            current = [target if z == source else z for z in current]
            used.add(source)
            actions.append({'source_Z': source, 'target_Z': target, 'source_symbol': self.meta[source]['symbol'],
                            'target_symbol': self.meta[target]['symbol'], 'radius_ratio': option['radius_ratio']})
        assert len(current) == len(old) and len(set(current)) == len(original_species)
        assert Counter(self.group(z) for z in old) == Counter(self.group(z) for z in current)
        known = all(z in self.meta and self.meta[z]['atomic_radius'] is not None for z in old + current)
        if known:
            ratio = (ordered_sum(self.meta[z]['atomic_radius'] for z in current)
                     / ordered_sum(self.meta[z]['atomic_radius'] for z in old))
            scale = max(.85, min(1.15, ratio))
            reason = 'radius_mean_ratio'
        else:
            scale = 1.
            reason = 'missing_radius_no_scaling'
        row = copy.deepcopy(original)
        row['atomic_numbers'] = current
        row['lattice'] = [[float(v) * scale for v in line] for line in original['lattice']]
        row['source'] = 'CrysFDiT-NFT plus ' + arm + ' chemical sampling'
        row['metadata'] = dict(row.get('metadata') or {}, chemical_sampling={
            'arm': arm, 'actions': actions, 'lattice_scale': scale, 'scale_reason': reason,
            'parent_id': original.get('id'), 'requested_max_edits': max_edits})
        assert all(math.isfinite(v) for line in row['lattice'] for v in line)
        return row
