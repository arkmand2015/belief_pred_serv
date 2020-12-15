import pickle
import os
import time
import numpy as np
from os import path
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from typing import NamedTuple
from collections import Counter
from transformers import pipeline
from async_calls import parallelize

MODEL = 0
WORD_MODEL = 1
CHAR_MODEL = 2
ZERO_MODEL = 'zero_shot_model'
CLASS_RESULT = 0
MULTICLASS_MODEL = 'multiclass'
DEFAULT_SUBCLASSES = ['ciencia', 'tecnologia', 'ingenieria', 'matematicas', 'gadgets', 'astrologia', 'esoterismo']
STEM_SUBCLASSES = ['ciencia', 'tecnologia', 'ingenieria', 'matematicas', 'gadgets']
ASTROLOGIA_SUBCLASSES = ['astrologia', 'esoterismo']
DEFAULT_SCORES = [0.0]*7
ZERO_SHOT_REPLACEMENT_CLASSES = {
'futbol soccer':'futbol', 'hogar e interiores':'hogar e indoors',
'juegos':'games', 'motores y vehiculos':'motores', 'pro-feminismo':'feminismo',
'aforismos':'sabiduria popular', 'vida al aire libre':'vida outdoors'
}


class ClassifierConfig(NamedTuple):
    text_col: str


class BeliefClassifier:

    def __init__(self):
        self.models = {}
        self.zero_shot_cat = [
        'arte y diseño', 'astrologia', 'esoterismo',
        'ciencia', 'tecnologia', 'ingenieria', 'matematicas', 'gadgets',
        'cine y tv', 'construccion', 'fiesta y vida nocturna',
        'finanzas e inversion', 'futbol soccer', 'gastronomia', 'golf',
        'historia y filosofia', 'hogar e interiores', 'juegos',
        'literatura y lectura', 'motores y vehiculos', 'musica',
        'noticias y actualidad', 'politica', 'aforismos',
        'salud', 'turismo y viajes', 'vida al aire libre'
        ]
        self.multiclass_cat = [
        'animales y mascotas', 'anti-corrupcion', 'anti-globalista',
        'capitalismo', 'deportes', 'ejercicio fisico',
        'globalista', 'justicia', 'militarista', 'nacionalista',
        'negocios y emprendimiento', 'other', 'pro-china',
        'pro-empresa privada', 'pro-ongs', 'romanticismo'
        ]
        self.bin_cat = [
        'anti-capitalismo', 'anti-china', 'anti-cuba',
        'anti-derecha', 'anti-eeuu', 'anti-empresa privada', 'anti-foro sao paulo',
        'anti-gloria porras', 'anti-gobierno guatemala',
        'anti-izquierda', 'anti-militarista', 'anti-nacionalista',
        'anti-ongs', 'anti-progre', 'anti-indigenista', 'comics y ciencia ficcion',
        'conservador', 'derecha', 'derechos humanos', 'humor', 'indigenista',
        'izquierda', 'pro-cuba', 'pro-eeuu', 'pro-gloria porras', 'pro-gobierno guatemala',
        'pro-foro sao paulo', 'progre', 'psicologia', 'racional',
        'sensibilidad social'
        ]
        self.tri_cat = [
        'trinary_feminismo', 'trinary_hellen'
        ]
        self.quat_cat = [
        'cuatrinary_cristianism'
        ]
        self.models_dir = Path(path.dirname(path.abspath(__file__)), 'models')

        self.reply_prob_gt80 = [
        'izquierda', 'anti-cristiano', 'memes', 'progre', 'anti-gobierno guatemala',
        'deportes', 'indigenista', 'derechos humanos', 'anti-derecha', 'motores y vehiculos', 'derecha',
        'psicologia', 'nacionalista', 'pro-empresa privada', 'pro-eeuu', 'pro-ongs', 'ejercicio fisico',
        'anti-globalista', 'comics y ciencia ficcion', 'pro-gobierno guatemala', 'anti-hellen mack',
        'pro-hellen mack', 'anti-gloria porras', 'pro-gloria porras', 'pro-foro sao paulo', 'conservador',
        'negocios y emprendimiento', 'futbol soccer', 'vida al aire libre', 'stem y gadgets', 'turismo y viajes',
        'animales y mascotas', 'juegos', 'astrologia y esoterismo', 'construccion', 'militarista',
        'capitalismo', 'pro-china', 'golf', 'anti-progre', 'anti-indigenista', 'anti-feminismo',
        'pro_feminismo', 'anti-nacionalista', 'anti-ongs', 'anti-foro sao paulo', 'anti-militarista',
        'anti-china', 'anti-capitalismo', 'anti-cuba'
        ]
        self.reply_prob_gt60 = [
        'evangelico', 'cristiano', 'familiar', 'anti-empresa privada',
        'pro-cuba', 'anti-izquierda', 'anti-eeuu', 'anti-corrupcion'
        ]
        self.reply_prob_gt95 = [
        'politica', 'humor', 'romanticismo', 'sabiduria_popular', 'sensibilidad social',
        'noticias y actualidad', 'justicia', 'musica', 'racional', 'globalista', 'gastronomia',
        'arte y diseño', 'salud', 'finanzas e inversion', 'hogar e interiores', 'literatura y lectura',
        'cine y tv', 'historia y filosofia', 'fiesta y vida nocturna', 'anti_negocios y emprendimiento'
        ]
        self.bio_prob_suff = [
        'memes', 'familiar', 'progre', 'anti-empresa privada', 'anti-gobierno guatemala', 'deportes', 'indigenista',
        'derechos humanos', 'anti-derecha', 'pro-cuba', 'anti-izquierda', 'motores y vehiculos', 'anti-eeuu', 'derecha',
        'psicologia', 'pro_feminismo', 'anti-corrupcion', 'nacionalista', 'pro-empresa privada', 'pro-eeuu',
        'pro-ongs', 'anti-globalista', 'comics y ciencia ficcion', 'gastronomia', 'pro-gobierno guatemala',
        'anti-hellen mack', 'pro-hellen mack', 'anti-gloria porras', 'pro-gloria porras', 'pro-foro sao paulo',
        'conservador', 'negocios y emprendimiento', 'futbol soccer', 'vida al aire libre', 'animales y mascotas', 'juegos',
        'literatura y lectura', 'astrologia y esoterismo', 'construccion', 'militarista', 'capitalismo', 'pro-china', 'golf',
        'anti-progre', 'anti-indigenista', 'anti-feminismo', 'anti-nacionalista', 'anti-ongs', 'anti-foro sao paulo',
        'anti-militarista', 'anti-china', 'anti-capitalismo', 'anti-cuba'
        ]
        self.timeline_suff_05 = [
        'anti-cristiano', 'familiar', 'anti-empresa privada', 'anti-gobierno guatemala', 'anti-derecha', 'pro-cuba',
        'anti-izquierda','anti-eeuu', 'anti-hellen mack', 'pro-hellen mack', 'anti-gloria porras', 'pro-gloria porras',
        'prochina', 'antiprogre'
        ]
        self.timeline_suff_620 = [
        'evangelico', 'cristiano', 'anti-cristiano', 'familiar',
        'progre', 'anti-empresa privada', 'anti-gobierno guatemala', 'indigenista', 'anti-derecha', 'pro-cuba', 'anti-izquierda',
        'anti-eeuu', 'anti-corrupcion', 'pro-empresa privada', 'pro-eeuu', 'pro-ongs', 'anti-globalista', 'comicsycienciaficcion',
        'pro-gobierno guatemala', 'anti-hellen mack', 'pro-hellen mack', 'anti-gloria porras', 'pro-gloria porras',
        'pro-foro sao paulo', 'conservador', 'vida al aire libre', 'animales y mascotas', 'construccion', 'militarista', 'capitalismo',
        'pro-china', 'golf', 'anti-progre', 'anti-indigenista', 'anti-feminismo', 'pro_feminismo', 'anti-nacionalista',
        'anti-ongs', 'anti-foro sao paulo', 'anti-militarista', 'anti-china', 'anti-capitalismo', 'anti-cuba'
        ]

    def get_supported_binary_beliefs(self):
        return self.bin_cat

    def get_supported_ternary_beliefs(self):
        return self.tri_cat

    def get_supported_quaternary_beliefs(self):
        return self.quat_cat

    def get_supported_multiclass_beliefs(self):
        return self.multiclass_cat

    def get_supported_zero_shot_beliefs(self):
        return self.zero_shot_cat

    def get_all_supported_beliefs(self):
        return self.get_supported_binary_beliefs() + self.get_supported_multiclass_beliefs() + self.get_supported_zero_shot_beliefs() + self.get_supported_ternary_beliefs() + self.get_supported_quaternary_beliefs()

    def load_models(self):
        non_zero_supported_cats = self.get_supported_binary_beliefs() + self.get_supported_ternary_beliefs() + self.get_supported_quaternary_beliefs() + [MULTICLASS_MODEL]
        for cat in non_zero_supported_cats:
            path_word = Path(self.models_dir, 'Word_model' + cat)
            path_char = Path(self.models_dir, 'char_model' + cat)
            path_model = Path(self.models_dir, cat)
            word_model = pickle.load(open(path_word, 'rb'))
            char_model = pickle.load(open(path_char, 'rb'))
            model = pickle.load(open(path_model, 'rb'))
            self.models[cat] = (model, word_model, char_model)
        zero_shot_model_path = Path(self.models_dir, ZERO_MODEL)
        if not os.path.isfile(zero_shot_model_path):
            model = pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli', device=0)
            pickle.dump(model, open(zero_shot_model_path, 'wb'))
        zero_shot_model = pickle.load(open(zero_shot_model_path, 'rb'))
        self.models[ZERO_MODEL] = (zero_shot_model)

    def predict(self, users_batch: list, zero_shot_enabled: bool):
        results = []
        for user in users_batch:
            user_predictions = {
            "user_id":user['user_id'],
            "labels":[]
            }
            hypothesis_template = 'This text is about {}.'
            non_zero_supported_cats = self.get_supported_binary_beliefs() + self.get_supported_ternary_beliefs() + self.get_supported_quaternary_beliefs() + [MULTICLASS_MODEL]
            if zero_shot_enabled:
                reply_probs = self.models[ZERO_MODEL](user['reply'], self.zero_shot_cat, hypothesis_template=hypothesis_template, multi_class=True)
            else:
                reply_probs = {"labels":DEFAULT_SUBCLASSES[:], "scores":DEFAULT_SCORES[:]}
            for cat in non_zero_supported_cats:
                df = pd.DataFrame(data={'text': [str(user['reply'])]})
                tfidf = sparse.hstack([self.models[cat][WORD_MODEL].transform(df['text']), self.models[cat][CHAR_MODEL].transform(df['text'])])
                cat_probs = self.models[cat][MODEL].predict_proba(tfidf)
                if cat is 'multiclass':
                  for index, multi_cat in enumerate(self.multiclass_cat):
                    reply_probs['labels'].append(multi_cat)
                    reply_probs['scores'].append(cat_probs[0][index])
                elif cat in self.get_supported_binary_beliefs():
                    reply_probs['labels'].append(cat)
                    reply_probs['scores'].append(cat_probs[0][1])
                elif cat in self.get_supported_ternary_beliefs():
                    reply_probs['labels'].append(f'anti-{cat}')
                    reply_probs['scores'].append(cat_probs[0][1])
                    reply_probs['labels'].append(f'pro-{cat}')
                    reply_probs['scores'].append(cat_probs[0][2])
                elif cat in self.get_supported_quaternary_beliefs():
                    reply_probs['labels'].append(f'anti-cristiano')
                    reply_probs['scores'].append(cat_probs[0][1])
                    reply_probs['labels'].append(f'cristiano')
                    reply_probs['scores'].append(cat_probs[0][2])
                    reply_probs['labels'].append(f'evangelico')
                    reply_probs['scores'].append(cat_probs[0][3])
                else:
                    reply_probs['labels'].append(cat)
                    reply_probs['scores'].append(cat_probs[CLASS_RESULT])
            
            reply_probs['labels'].append('stem y gadgets')
            maximum = 0.0
            for cat in STEM_SUBCLASSES:
                current_cat = reply_probs['scores'][reply_probs['labels'].index(cat)]
                if current_cat > maximum:
                    maximum = current_cat
            reply_probs['scores'].append(maximum)

            reply_probs['labels'].append('astrologia y esoterismo')

            maximum = 0.0
            for cat in ASTROLOGIA_SUBCLASSES:
                current_cat = reply_probs['scores'][reply_probs['labels'].index(cat)]
                if current_cat > maximum:
                    maximum = current_cat
            reply_probs['scores'].append(maximum)
            for cat in DEFAULT_SUBCLASSES:
                remove_index = reply_probs['labels'].index(cat)
                reply_probs['labels'].pop(remove_index)
                reply_probs['scores'].pop(remove_index)
            for index, score in enumerate(reply_probs['scores']):
                if score >= 0.95 and reply_probs['labels'][index] in self.reply_prob_gt95:
                    user_predictions['labels'].append(reply_probs['labels'][index])
                if score >= 0.80 and reply_probs['labels'][index] in self.reply_prob_gt80:
                    user_predictions['labels'].append(reply_probs['labels'][index])
                if score >= 0.60 and reply_probs['labels'][index] in self.reply_prob_gt60:
                    user_predictions['labels'].append(reply_probs['labels'][index])

            if zero_shot_enabled and 'bio' in user and user['bio']:
                bio_probs = self.models[ZERO_MODEL](user['bio'], self.zero_shot_cat, hypothesis_template=hypothesis_template, multi_class=True)
            else:
                bio_probs = {"labels":DEFAULT_SUBCLASSES[:], "scores":DEFAULT_SCORES[:]}
            for cat in non_zero_supported_cats:
                df = pd.DataFrame(data={'text': [str(user['bio'])]})
                tfidf = sparse.hstack([self.models[cat][WORD_MODEL].transform(df['text']), self.models[cat][CHAR_MODEL].transform(df['text'])])
                cat_probs = self.models[cat][MODEL].predict_proba(tfidf)
                if cat is 'multiclass':
                  for index, multi_cat in enumerate(self.multiclass_cat):
                    bio_probs['labels'].append(multi_cat)
                    bio_probs['scores'].append(cat_probs[0][index])
                elif cat in self.get_supported_binary_beliefs():
                    bio_probs['labels'].append(cat)
                    bio_probs['scores'].append(cat_probs[0][1])
                elif cat in self.get_supported_ternary_beliefs():
                    bio_probs['labels'].append(f'anti-{cat}')
                    bio_probs['scores'].append(cat_probs[0][1])
                    bio_probs['labels'].append(f'pro-{cat}')
                    bio_probs['scores'].append(cat_probs[0][2])
                elif cat in self.get_supported_quaternary_beliefs():
                    bio_probs['labels'].append(f'anti-cristiano')
                    bio_probs['scores'].append(cat_probs[0][1])
                    bio_probs['labels'].append(f'cristiano')
                    bio_probs['scores'].append(cat_probs[0][2])
                    bio_probs['labels'].append(f'evangelico')
                    bio_probs['scores'].append(cat_probs[0][3])
                else:
                    bio_probs['labels'].append(cat)
                    bio_probs['scores'].append(cat_probs[CLASS_RESULT])

            bio_probs['labels'].append('stem y gadgets')

            maximum = 0.0
            for cat in STEM_SUBCLASSES:
                current_cat = bio_probs['scores'][bio_probs['labels'].index(cat)]
                if current_cat > maximum:
                    maximum = current_cat
            bio_probs['scores'].append(maximum)

            bio_probs['labels'].append('astrologia y esoterismo')

            maximum = 0.0
            for cat in ASTROLOGIA_SUBCLASSES:
                current_cat = bio_probs['scores'][bio_probs['labels'].index(cat)]
                if current_cat > maximum:
                    maximum = current_cat
            bio_probs['scores'].append(maximum)

            for cat in DEFAULT_SUBCLASSES:
                remove_index = bio_probs['labels'].index(cat)
                bio_probs['labels'].pop(remove_index)
                bio_probs['scores'].pop(remove_index)

            for index, score in enumerate(bio_probs['scores']):
                if score >= 0.90 and bio_probs['labels'][index] in self.bio_prob_suff:
                    user_predictions['labels'].append(bio_probs['labels'][index])

            timeline_cats = []
            if zero_shot_enabled:
                tweet_zero_responses = parallelize(user['user_id'], user['timeline'])
                tweet_probs_zero = [pred for response in tweet_zero_responses for pred in response.json()['prediction']]
              
            for index, tweet in enumerate(user['timeline']):
                if zero_shot_enabled:
                    tweet_probs = {'labels':tweet_probs_zero[index]['labels'], 'scores':tweet_probs_zero[index]['scores']}
                else:
                    tweet_probs = {'labels':DEFAULT_SUBCLASSES[:], 'scores':DEFAULT_SCORES[:]}
                for cat in non_zero_supported_cats:
                    df = pd.DataFrame(data={'text': [str(tweet)]})
                    tfidf = sparse.hstack([self.models[cat][WORD_MODEL].transform(df['text']), self.models[cat][CHAR_MODEL].transform(df['text'])])
                    cat_probs = self.models[cat][MODEL].predict_proba(tfidf)
                    if cat is 'multiclass':
                      for index, multi_cat in enumerate(self.multiclass_cat):
                        tweet_probs['labels'].append(multi_cat)
                        tweet_probs['scores'].append(cat_probs[0][index])
                    elif cat in self.get_supported_binary_beliefs():
                        tweet_probs['labels'].append(cat)
                        tweet_probs['scores'].append(cat_probs[0][1])
                    elif cat in self.get_supported_ternary_beliefs():
                        tweet_probs['labels'].append(f'anti-{cat}')
                        tweet_probs['scores'].append(cat_probs[0][1])
                        tweet_probs['labels'].append(f'pro-{cat}')
                        tweet_probs['scores'].append(cat_probs[0][2])
                    elif cat in self.get_supported_quaternary_beliefs():
                        tweet_probs['labels'].append(f'anti-cristiano')
                        tweet_probs['scores'].append(cat_probs[0][1])
                        tweet_probs['labels'].append(f'cristiano')
                        tweet_probs['scores'].append(cat_probs[0][2])
                        tweet_probs['labels'].append(f'evangelico')
                        tweet_probs['scores'].append(cat_probs[0][3])
                    else:
                        tweet_probs['labels'].append(cat)
                        tweet_probs['scores'].append(cat_probs[CLASS_RESULT])

                tweet_probs['labels'].append('stem y gadgets')

                maximum = 0.0
                for cat in STEM_SUBCLASSES:
                    current_cat = tweet_probs['scores'][tweet_probs['labels'].index(cat)]
                    if current_cat > maximum:
                        maximum = current_cat
                tweet_probs['scores'].append(maximum)

                tweet_probs['labels'].append('astrologia y esoterismo')

                maximum = 0.0
                for cat in ASTROLOGIA_SUBCLASSES:
                    current_cat = tweet_probs['scores'][tweet_probs['labels'].index(cat)]
                    if current_cat > maximum:
                        maximum = current_cat
                tweet_probs['scores'].append(maximum)

                for cat in DEFAULT_SUBCLASSES:
                    remove_index = tweet_probs['labels'].index(cat)
                    tweet_probs['labels'].pop(remove_index)
                    tweet_probs['scores'].pop(remove_index)

                for index, score in enumerate(tweet_probs['scores']):
                    if score >= 0.90:
                        timeline_cats.append(tweet_probs['labels'][index])
            cat_tweet_count = Counter(timeline_cats)
            cat_tweet_total = sum(cat_tweet_count.values())
            cat_tweet_ratio = {cat: count / cat_tweet_total for cat, count in cat_tweet_count.items()}
            user_predictions['labels'] += [cat for cat, timeline_ratio in cat_tweet_ratio.items() if timeline_ratio >= 0.20]
            user_predictions['labels'] += [cat for cat, timeline_ratio in cat_tweet_ratio.items() if (timeline_ratio >= 0.06 and cat in self.timeline_suff_620)]
            user_predictions['labels'] += [cat for cat, timeline_ratio in cat_tweet_ratio.items() if (timeline_ratio >= 0.05 and cat in self.timeline_suff_05)]
            user_predictions['labels'] = list(set(user_predictions['labels']))
            user_predictions['labels'] = [ZERO_SHOT_REPLACEMENT_CLASSES[label] if label in ZERO_SHOT_REPLACEMENT_CLASSES else label for label in user_predictions['labels']]
            results.append(user_predictions)
        return results
