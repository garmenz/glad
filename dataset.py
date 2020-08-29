import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from stanza.nlp.corenlp import CoreNLPClient
from collections import namedtuple

client = None

EvalResult = namedtuple(typename='EvalResult', field_names=['turn_request', 'turn_inform', 'joint_goal'])


def annotate(sent):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words = []
    for sent in client.annotate(sent).sentences:
        for tok in sent:
            words.append(tok.word)
    return words


class Turn:

    def __init__(self, transcript, turn_label, system_acts, system_transcript, num=None):
        self.transcript = transcript
        self.turn_label = turn_label
        self.system_acts = system_acts
        self.system_transcript = system_transcript
        self.num = num or {}

    # def to_dict(self):
    #     return {'turn_id': self.id, 'transcript': self.transcript, 'turn_label': self.turn_label, 'belief_state': self.belief_state, 'system_acts': self.system_acts, 'system_transcript': self.system_transcript, 'num': self.num}
    def to_dict(self):
        return {'transcript': self.transcript, 'turn_label': self.turn_label, 'system_acts': self.system_acts, 'system_transcript': self.system_transcript, 'num': self.num}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def annotate_raw(cls, raw):

        return cls(
            transcript=annotate(raw['patient_text']),
            system_acts=raw['system_acts'],
            turn_label=raw['turn_label'],
            system_transcript=raw['doctor_text'],
        )

    def numericalize_(self, vocab):
        self.num['transcript'] = vocab.word2index(['<sos>'] + [w.lower() for w in self.transcript + ['<eos>']], train=True)
        self.num['system_acts'] = [vocab.word2index(['<sos>'] + [w.lower() for w in a] + ['<eos>'], train=True) for a in self.system_acts + [['<sentinel>']]]


class Dialogue:

    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {'dialogue_index': self.id, 'turns': [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_index'], [Turn.from_dict(t) for t in d['turns']])

    @classmethod
    def annotate_raw(cls, raw):
        return cls(raw['dialogue_index'], [Turn.annotate_raw(t) for t in raw['turns']])


class Dataset:

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_turns(self):
        for d in self.dialogues:
            for t in d.turns:
                yield t

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d):
        return cls([Dialogue.from_dict(dd) for dd in d['dialogues']])

    @classmethod
    def annotate_raw(cls, fname):
        with open(fname) as f:
            data = json.load(f)
            return cls([Dialogue.annotate_raw(d) for d in tqdm(data)])

    def numericalize_(self, vocab):
        for t in self.iter_turns():
            t.numericalize_(vocab)


    # used by WOZ dataset
    # def extract_ontology(self):
    #     slots = set()
    #     values = defaultdict(set)
    #     for t in self.iter_turns():
    #         for s, v in t.turn_label:
    #             slots.add(s.lower())
    #             values[s].add(v.lower())
    #     return Ontology(sorted(list(slots)), {k: sorted(list(v)) for k, v in values.items()})

    def batch(self, batch_size, shuffle=False):
        turns = list(self.iter_turns())
        if shuffle:
            np.random.shuffle(turns)
        for i in tqdm(range(0, len(turns), batch_size)):
            yield turns[i:i+batch_size]

    def evaluate_preds(self, preds):
        joint_goal = []
        turn_goal = []
        i = 0
        for d in self.dialogues:
            pred_state, gold_state = {}, {}
            for t in d.turns:
                gold_inform = set([(s, v) for s, v in t.turn_label])
                pred_inform = set([(s, v) for s, v in preds[i]])
                turn_goal.append(gold_inform == pred_inform)
                for s, v in pred_inform:
                    pred_state[s] = v
                for s, v in gold_inform:
                    gold_state[s] = v
                joint_goal.append(pred_state == gold_state)
        return {'joint_goal': np.mean(joint_goal), 'turn_goal': np.mean(turn_goal)}



    def record_preds(self, preds, to_file):
        data = self.to_dict()
        i = 0
        for d in data['dialogues']:
            for t in d['turns']:
                t['pred'] = sorted(list(preds[i]))
                i += 1
        with open(to_file, 'wt') as f:
            json.dump(data, f)


class Ontology:

    def __init__(self, slots=None, values=None, num=None):
        self.slots = slots or []
        self.values = values or {}
        self.num = num or {}

    def __add__(self, another):
        new_slots = sorted(list(set(self.slots + another.slots)))
        new_values = {s: sorted(list(set(self.values.get(s, []) + another.values.get(s, [])))) for s in new_slots}
        return Ontology(new_slots, new_values)

    def __radd__(self, another):
        return self if another == 0 else self.__add__(another)

    def to_dict(self):
        return {'slots': self.slots, 'values': self.values, 'num': self.num}

    def numericalize_(self, vocab):
        self.num = {}
        for s, vs in self.values.items():
            self.num[s] = [vocab.word2index(annotate('{} = {}'.format(s, v)) + ['<eos>'], train=True) for v in vs]

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
