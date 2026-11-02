"""Resolve plain poisoned corpora and self-contained attack artifacts."""
from .attacks import canonical_method, intent
from .io import normalize_documents


def resolve_attack(payload, method=None, target=None, attacker_intent=None):
    artifact = isinstance(payload, dict) and payload.get('artifact_version') == 'ars-attack/1'
    if isinstance(payload, dict) and not artifact and 'documents' not in payload:
        raise ValueError('--attack-documents must contain document rows or an ars-attack/1 artifact')
    documents = normalize_documents(payload)
    if not documents:
        raise ValueError('Attack document collection is empty')

    def choose(explicit, name, flag, transform=lambda x: x):
        if explicit is not None:
            value = str(explicit).strip()
            if not value:
                raise ValueError(f'{flag} must not be empty')
            return transform(value)
        if artifact and payload.get(name):
            return transform(str(payload[name]))
        values = {transform(str(d['metadata'][name])) for d in documents
                  if d['metadata'].get(name) not in (None, '')}
        if len(values) != 1:
            reason = 'conflicting' if values else 'missing'
            raise ValueError(f'Attack document metadata has {reason} {name}; supply {flag}')
        return values.pop()

    method = choose(method, 'attack_method', '--attack', canonical_method)
    target = choose(target, 'target_answer', '--target-answer')
    if attacker_intent is None:
        attacker_intent = (payload.get('attacker_intent') if artifact else None) or intent(method, target)
        if artifact and (method != payload.get('attack_method') or target != payload.get('target_answer')):
            attacker_intent = intent(method, target)
    if not str(attacker_intent).strip():
        raise ValueError('--attacker-intent must not be empty')
    return {'documents': documents, 'attack_method': method, 'target_answer': target,
            'attacker_intent': attacker_intent,
            'clean': payload.get('clean') if artifact else None,
            'clean_experiment': payload.get('clean_experiment', {}) if artifact else {}}
