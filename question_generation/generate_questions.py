'''
Based on:
clevr-dataset-gen
https://github.com/facebookresearch/clevr-dataset-gen
'''

from __future__ import print_function
import argparse
import json
import os
import random
import time
import re
from timeit import default_timer as timer
import copy

import question_engine as qeng

"""
Quetion generation for SHOP-VRB
"""

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument('--input_scene_file', default='../output/SHOP_VRB_scenes.json',
                    help="JSON file containing ground-truth scene information" +
                    " for all images from render_images.py")
parser.add_argument('--metadata_file', default='metadata_shop_vrb.json',
                    help="JSON file containing metadata about functions")
parser.add_argument('--synonyms_json', default='synonyms.json',
                    help="JSON file defining synonyms for parameter values")
parser.add_argument('--plurals_json', default='plurals.json',
                    help="JSON file defining plurals for parameter values")
parser.add_argument('--phrasings_json', default='phrasings.json',
                    help="JSON file defining some specific phrases to change")
parser.add_argument('--template_dir', default='SHOP_VRB_templates',
                    help="Directory containing JSON templates for questions")

# Output
parser.add_argument('--output_questions_file',
                    default='../output/SHOP_VRB_questions.json',
                    help="The output file to write containing generated questions")
parser.add_argument('--dump_output_every', default=0, type=int,
                    help="Split question file into bits corresponding to n scenes. " +
                    "Helps with estimating memory usage throughout the process." +
                    "Zero means no splitting.")

# Control which and how many images to process
parser.add_argument('--scene_start_idx', default=0, type=int,
                    help="The image at which to start generating questions;" +
                    " this allows question generation to be split across many workers")
parser.add_argument('--num_scenes', default=0, type=int,
                    help="The number of images for which to generate questions." +
                    "Setting to 0 generates questions for all scenes in the " +
                    "input file starting from --scene_start_idx")

# Control the number of questions per image; we will attempt to generate
# templates_per_image * instances_per_template questions per image.
parser.add_argument('--templates_per_image', default=10, type=int,
                    help="The number of different templates that should be " +
                    "instantiated on each image")
parser.add_argument('--instances_per_template', default=1, type=int,
                    help="The number of times each template should be " +
                    "instantiated on an image")

# Misc
parser.add_argument('--reset_counts_every', default=250, type=int,
                    help="How often to reset template and answer counts. " +
                    "Higher values will result in flatter distributions " +
                    "over templates and answers, but will result in longer runtimes.")
parser.add_argument('--verbose', action='store_true',
                    help="Print more verbose output")
parser.add_argument('--time_dfs', action='store_true',
                    help="Time each depth-first search; must be given with --verbose")
parser.add_argument('--profile', action='store_true',
                    help="If given then run inside cProfile")
parser.add_argument('--timeout', default=60, type=int,
                    help="Skip templete if failed to generate for n seconds")
# args = parser.parse_args()


def precompute_filter_options(scene_struct, metadata, visual_uniqueness=False):
    # Keys are tuples (size, color, shape, material) (where some may be None)
    # and values are lists of object idxs that match the filter criterion
    attribute_map = {}
    vis_key = '_filter_options'

    if metadata['dataset'] == 'SHOP-VRB':
        attr_keys = ['size', 'weight', 'color', 'material', 'movability', 'shape', 'name']
    elif metadata['dataset'] == 'SHOP-VRB-text':
        attr_keys = ['size', 'weight', 'color', 'material', 'movability', 'shape', 'name',
                     'powering', 'disassembly', 'picking', 'attribute']
    else:
        assert False, 'Unrecognized dataset'

    # Precompute masks
    masks = []
    for i in range(2 ** len(attr_keys)):
        mask = []
        for j in range(len(attr_keys)):
            mask.append((i // (2 ** j)) % 2)
        masks.append(mask)

    for object_idx, obj in enumerate(scene_struct['objects']):
        if metadata['dataset'] == 'SHOP-VRB':
            keys = [tuple(obj[k] for k in attr_keys)]
        elif metadata['dataset'] == 'SHOP-VRB-text':
            keys = [tuple(obj[k] for k in attr_keys)]
        for mask in masks:
            for key in keys:
                masked_key = []
                for a, b in zip(key, mask):
                    if b == 1:
                        if isinstance(a, list):
                            masked_keys = []
                            for i in range(len(a)):
                                masked_keys.append(masked_key[:])
                                # print(masked_key[i])
                                masked_keys[i].append(a[i])
                            masked_key = masked_keys
                        else:
                            masked_key.append(a)
                    else:
                        masked_key.append(None)
                if isinstance(masked_key[0], list):
                    # print(masked_key)
                    for m_key in masked_key:
                        m_key = tuple(m_key)
                        if m_key not in attribute_map:
                            attribute_map[m_key] = set()
                        attribute_map[m_key].add(object_idx)
                else:
                    masked_key = tuple(masked_key)
                    if masked_key not in attribute_map:
                        attribute_map[masked_key] = set()
                    attribute_map[masked_key].add(object_idx)
    scene_struct[vis_key] = attribute_map


def find_filter_options(object_idxs, scene_struct, metadata, visual_uniqueness=False):
    # Keys are tuples (where some may be None)
    # and values are lists of object idxs that match the filter criterion

    if visual_uniqueness:
        vis_key = '_filter_options_visual'
    else:
        vis_key = '_filter_options'

    if vis_key not in scene_struct:
        precompute_filter_options(scene_struct, metadata, visual_uniqueness)

    attribute_map = {}
    object_idxs = set(object_idxs)
    for k, vs in scene_struct[vis_key].items():
        attribute_map[k] = sorted(list(object_idxs & vs))
    return attribute_map


def add_empty_filter_options(attribute_map, metadata, num_to_add):
    # Add some filtering criterion that do NOT correspond to objects

    if metadata['dataset'] == 'SHOP-VRB':
        attr_keys = ['Size', 'Weight', 'Color', 'Material', 'Movability', 'Shape', 'Name']
    elif metadata['dataset'] == 'SHOP-VRB-text':
        attr_keys = ['Size', 'Weight', 'Color', 'Material', 'Movability', 'Shape', 'Name',
                     'Powering', 'Disassembly', 'Picking', 'Attribute']
    else:
        assert False, 'Unrecognized dataset'

    attr_vals = [metadata['types'][t] + [None] for t in attr_keys]
    if '_filter_options' in metadata:
        attr_vals = metadata['_filter_options']

    target_size = len(attribute_map) + num_to_add
    while len(attribute_map) < target_size:
        k = (random.choice(v) for v in attr_vals)
        if k not in attribute_map:
            attribute_map[k] = []


def find_relate_filter_options(object_idx, scene_struct, metadata,
                               unique=False, include_zero=False, trivial_frac=0.1):
    options = {}
    if '_filter_options' not in scene_struct:
        precompute_filter_options(scene_struct, metadata)

    trivial_options = {}
    for relationship in scene_struct['relationships']:
        related = set(scene_struct['relationships'][relationship][object_idx])
        for filters, filtered in scene_struct['_filter_options'].items():
            intersection = related & filtered
            trivial = (intersection == filtered)
            if unique and len(intersection) != 1:
                continue
            if not include_zero and len(intersection) == 0:
                continue
            if trivial:
                trivial_options[(relationship, filters)] = sorted(list(intersection))
            else:
                options[(relationship, filters)] = sorted(list(intersection))

    N, f = len(options), trivial_frac
    num_trivial = int(round(N * f / (1 - f)))
    trivial_options = list(trivial_options.items())
    random.shuffle(trivial_options)
    for k, v in trivial_options[:num_trivial]:
        options[k] = v

    return options


def node_shallow_copy(node):
    new_node = {
        'function': node['function'],
        'inputs': node['inputs'],
    }
    if 'value_inputs' in node:
        new_node['value_inputs'] = node['value_inputs']
    else:
        new_node['value_inputs'] = []
    return new_node


def other_heuristic(text, param_vals, metadata):
    """
    Post-processing heuristic to handle the word "other"
    """
    if ' other ' not in text and ' another ' not in text:
        return text

    if metadata['dataset'] == 'SHOP-VRB':
        target_keys = {
            '<Z>', '<W>', '<C>', '<M>', '<F>', '<S>', '<N>',
            '<Z2>', '<W2>', '<C2>', '<M2>', '<F2>', '<S2>', '<N2>',
        }
        key_pairs = [
            ('<Z>', '<Z2>'),
            ('<W>', '<W2>'),
            ('<C>', '<C2>'),
            ('<M>', '<M2>'),
            ('<F>', '<F2>'),
            ('<S>', '<S2>'),
            ('<N>', '<N2>'),
        ]
    elif metadata['dataset'] == 'SHOP-VRB-text':
        target_keys = {
            '<Z>', '<W>', '<C>', '<M>', '<F>', '<S>', '<N>', '<P>', '<D>', '<K>', '<A>',
            '<Z2>', '<W2>', '<C2>', '<M2>', '<F2>', '<S2>', '<N2>', '<P2>', '<D2>', '<K2>', '<A2>',
        }
        key_pairs = [
            ('<Z>', '<Z2>'),
            ('<W>', '<W2>'),
            ('<C>', '<C2>'),
            ('<M>', '<M2>'),
            ('<F>', '<F2>'),
            ('<S>', '<S2>'),
            ('<N>', '<N2>'),
            ('<P>', '<P2>'),
            ('<D>', '<D2>'),
            ('<A>', '<A2>'),
            ('<K>', '<K2>'),
        ]
    else:
        assert False, 'Unrecognized dataset'

    if param_vals.keys() != target_keys:
        return text

    remove_other = False
    for k1, k2 in key_pairs:
        v1 = param_vals.get(k1, None)
        v2 = param_vals.get(k2, None)
        if v1 != '' and v2 != '' and v1 != v2:
            print('other has got to go! %s = %s but %s = %s'
                  % (k1, v1, k2, v2))
            remove_other = True
            break
    if remove_other:
        if ' other ' in text:
            text = text.replace(' other ', ' ')
        if ' another ' in text:
            text = text.replace(' another ', ' a ')
    return text


def instantiate_templates_dfs(scene_struct, template, metadata, answer_counts,
                              synonyms, plurals, phrasings, max_instances=None, verbose=False):

    param_name_to_type = {p['name']: p['type'] for p in template['params']}

    initial_state = {
        'nodes': [node_shallow_copy(template['nodes'][0])],
        'vals': {},
        'input_map': {0: 0},
        'next_template_node': 1,
    }

    # Visual uniqueness constraint
    visual_uniqueness = False

    states = [initial_state]
    final_states = []
    start = timer()

    additional_constraints = []
    fix_constraints = []
    for constraint in template['constraints']:
        if constraint['type'] == 'NONE_OF':
            for c in constraint['params']:
                additional_constraints.append(
                    {
                        "params": [c],
                        "type": "NULL"
                    }
                )
        if constraint['type'] in ['ONE_OF', 'ONE_OF_FORCE']:
            new_constrs_idx = random.sample(range(len(constraint['params'])), len(constraint['params']))
            for idx in new_constrs_idx[:-1]:
                additional_constraints.append(
                    {
                        "params": [constraint['params'][idx]],
                        "type": "NULL"
                    }
                )
            if constraint['type'] == 'ONE_OF_FORCE':
                fix_constraints.append(constraint['params'][new_constrs_idx[-1]])
    while states:
        stop = timer()
        if stop - start > args.timeout:
            print("Timed out")
            break
        state = states.pop()
        # Check to make sure the current state is valid
        q = {'nodes': state['nodes']}
        outputs = qeng.answer_question(q, metadata, scene_struct, all_outputs=True)
        answer = outputs[-1]
        if answer == '__INVALID__':
            continue
        # Check to make sure constraints are satisfied for the current state
        skip_state = False
        for constraint in template['constraints'] + additional_constraints:
            if constraint['type'] == 'NEQ':
                p1, p2 = constraint['params']
                v1, v2 = state['vals'].get(p1), state['vals'].get(p2)
                if v1 is not None and v2 is not None and v1 != v2:
                    if verbose:
                        print('skipping due to NEQ constraint')
                        print(constraint)
                        print(state['vals'])
                    skip_state = True
                    break
            elif constraint['type'] == 'NULL':
                p = constraint['params'][0]
                p_type = param_name_to_type[p]
                v = state['vals'].get(p)
                if v is not None:
                    skip = False
                    if metadata['dataset'] in ['SHOP-VRB', 'SHOP-VRB-text']:
                        if p_type in ['Shape', 'Name'] and v not in ['thing', '']:
                            skip = True
                        if p_type not in ['Shape', 'Name'] and v != '':
                            skip = True
                    if skip:
                        if verbose:
                            print('skipping due to NULL constraint')
                            print(constraint)
                            print(state['vals'])
                        skip_state = True
                        break
            elif constraint['type'] == 'OUT_NEQ':
                i, j = constraint['params']
                i = state['input_map'].get(i, None)
                j = state['input_map'].get(j, None)
                if i is not None and j is not None and outputs[i] == outputs[j]:
                    if verbose:
                        print('skipping due to OUT_NEQ constraint')
                        print(outputs[i])
                        print(outputs[j])
                    skip_state = True
                    break
            elif constraint['type'] == 'VISUAL_UNIQUE':
                visual_uniqueness = True
            elif constraint['type'] == 'ONE_OF':
                pass
            elif constraint['type'] == 'NONE_OF':
                pass
            elif constraint['type'] == 'ONE_OF_FORCE':
                counter = 0
                counter_sum = 0
                for c in fix_constraints:
                    if c in state['vals']:
                        v = state['vals'].get(c)
                        counter_sum += 1
                        if v == '':
                            counter += 1
                if counter != 0 and counter_sum != 0:
                    skip_state = True
                    break
            else:
                assert False, 'Unrecognized constraint type "%s"' % constraint['type']

        if skip_state:
            continue

        # We have already checked to make sure the answer is valid, so if we have
        # processed all the nodes in the template then the current state is a valid
        # question, so add it if it passes our rejection sampling tests.
        # print(state['next_template_node'], len(template['nodes']))
        if state['next_template_node'] == len(template['nodes']):
            # Use our rejection sampling heuristics to decide whether we should
            # keep this template instantiation
            if isinstance(answer, list):
                if tuple(answer) not in answer_counts:
                    answer_counts[tuple(answer)] = 0
                cur_answer_count = answer_counts[tuple(answer)]
            else:
                cur_answer_count = answer_counts[answer]
            answer_counts_sorted = sorted(answer_counts.values())
            median_count = answer_counts_sorted[len(answer_counts_sorted) // 2]
            median_count = max(median_count, 5)
            if cur_answer_count > 1.1 * answer_counts_sorted[-2]:
                if verbose:
                    print('skipping due to second count')
                continue
            if cur_answer_count > 5.0 * median_count:
                if verbose:
                    print('skipping due to median')
                continue

            # If the template contains a raw relate node then we need to check for
            # degeneracy at the end
            has_relate = any(n['function'] == 'relate' for n in template['nodes'])
            if has_relate:
                degen = qeng.is_degenerate(q, metadata, scene_struct,
                                           answer=answer, verbose=verbose)
                if degen:
                    continue

            if isinstance(answer, list):
                answer_counts[tuple(answer)] += 1
            else:
                answer_counts[answer] += 1
            state['answer'] = answer
            final_states.append(state)
            if max_instances is not None and len(final_states) == max_instances:
                break
            continue

        # Otherwise fetch the next node from the template
        # Make a shallow copy so cached _outputs don't leak ... this is very nasty
        next_node = template['nodes'][state['next_template_node']]
        next_node = node_shallow_copy(next_node)

        special_nodes = {
            'filter_unique', 'filter_count', 'filter_exist', 'filter',
            'relate_filter', 'relate_filter_unique', 'relate_filter_count',
            'relate_filter_exist',
        }

        if next_node['function'] in special_nodes:
            if next_node['function'].startswith('relate_filter'):
                unique = (next_node['function'] == 'relate_filter_unique')
                include_zero = (next_node['function'] == 'relate_filter_count' or
                                next_node['function'] == 'relate_filter_exist')
                filter_options = find_relate_filter_options(answer, scene_struct, metadata,
                                                            unique=unique, include_zero=include_zero)
            else:
                filter_options = find_filter_options(answer, scene_struct, metadata)
                if next_node['function'] == 'filter':
                    # Remove null filter
                    filter_options.pop((None, None, None, None), None)
                    filter_options.pop((None, None, None, None, None, None, None, None), None)
                if next_node['function'] == 'filter_unique':
                    if visual_uniqueness:
                        filter_options = find_filter_options(answer, scene_struct, metadata, visual_uniqueness)
                    # Get rid of all filter options that don't result in a single object
                    filter_options = {k: v for k, v in filter_options.items() if len(v) == 1}
                else:
                    # Add some filter options that do NOT correspond to the scene
                    if next_node['function'] == 'filter_exist':
                        # For filter_exist we want an equal number that do and don't
                        num_to_add = len(filter_options)
                    elif next_node['function'] == 'filter_count' or next_node['function'] == 'filter':
                        # For filter_count add nulls equal to the number of singletons
                        num_to_add = sum(1 for k, v in filter_options.items() if len(v) == 1)
                    add_empty_filter_options(filter_options, metadata, num_to_add)

            filter_option_keys = list(filter_options.keys())
            random.shuffle(filter_option_keys)
            for k in filter_option_keys:
                # Remove filters not holding fixed constraints
                new_nodes = []
                cur_next_vals = {k: v for k, v in state['vals'].items()}
                next_input = state['input_map'][next_node['inputs'][0]]
                filter_side_inputs = next_node['value_inputs']
                if next_node['function'].startswith('relate'):
                    param_name = next_node['value_inputs'][0]  # First one should be relate
                    filter_side_inputs = next_node['value_inputs'][1:]
                    param_type = param_name_to_type[param_name]
                    assert param_type == 'Relation'
                    param_val = k[0]
                    k = k[1]
                    new_nodes.append({
                        'function': 'relate',
                        'inputs': [next_input],
                        'value_inputs': [param_val],
                    })
                    cur_next_vals[param_name] = param_val
                    next_input = len(state['nodes']) + len(new_nodes) - 1
                for param_name, param_val in zip(filter_side_inputs, k):
                    param_type = param_name_to_type[param_name]
                    filter_type = 'filter_%s' % param_type.lower()
                    # print(param_val)
                    if param_val is not None:
                        new_nodes.append({
                            'function': filter_type,
                            'inputs': [next_input],
                            'value_inputs': [param_val],
                        })
                        if metadata['dataset'] in ['SHOP-VRB', 'SHOP-VRB-text']:
                            pass
                            if param_type == 'Shape':
                                param_val = param_val + ' thing'

                        cur_next_vals[param_name] = param_val
                        next_input = len(state['nodes']) + len(new_nodes) - 1
                    elif param_val is None:
                        if metadata['dataset'] in ['SHOP-VRB', 'SHOP-VRB-text']:
                            constr = None
                            for constraint in template['constraints'] + additional_constraints:
                                if constraint['type'] not in ["VISUAL_UNIQUE"]:
                                    if param_name in constraint["params"] and constraint['type'] not in ['ONE_OF', 'ONE_OF_FORCE']:
                                        if 'N' in param_name:
                                            constr = 'Name'
                                            for constraint in template['constraints'] + additional_constraints:
                                                if param_name.replace('N', 'S') in constraint["params"] and constraint['type'] not in ['ONE_OF', 'ONE_OF_FORCE']:
                                                    constr = 'both'
                                        elif 'S' in param_name:
                                            constr = 'Shape'
                        if metadata['dataset'] in ['CLEVR-v1.0', 'CLEVR_text'] and param_type == 'Shape':
                            param_val = 'thing'
                        elif metadata['dataset'] in ['SHOP-VRB', 'SHOP-VRB-text']:
                            if param_type == 'Shape':
                                if constr == 'Shape':
                                    param_val = ''
                                else:
                                    param_val = 'thing'
                            elif param_type == 'Name':
                                if constr == 'Name':
                                    param_val = ''
                                else:
                                    param_val = 'thing'
                            else:
                                param_val = ''
                        else:
                            param_val = ''
                        cur_next_vals[param_name] = param_val
                input_map = {k: v for k, v in state['input_map'].items()}
                extra_type = None
                if next_node['function'].endswith('unique'):
                    extra_type = 'unique'
                if next_node['function'].endswith('count'):
                    extra_type = 'count'
                if next_node['function'].endswith('exist'):
                    extra_type = 'exist'
                if extra_type is not None:
                    new_nodes.append({
                        'function': extra_type,
                        'inputs': [input_map[next_node['inputs'][0]] + len(new_nodes)],
                        'value_inputs': []
                    })
                input_map[state['next_template_node']] = len(state['nodes']) + len(new_nodes) - 1
                states.append({
                    'nodes': state['nodes'] + new_nodes,
                    'vals': cur_next_vals,
                    'input_map': input_map,
                    'next_template_node': state['next_template_node'] + 1,
                })

        elif len(next_node['value_inputs']) > 0:
            # If the next node has template parameters, expand them out
            assert len(next_node['value_inputs']) == 1, 'NOT IMPLEMENTED'

            # Use metadata to figure out domain of valid values for this parameter.
            # Iterate over the values in a random order; then it is safe to bail
            # from the DFS as soon as we find the desired number of valid template
            # instantiations.
            param_name = next_node['value_inputs'][0]
            param_type = param_name_to_type[param_name]
            param_vals = metadata['types'][param_type][:]
            random.shuffle(param_vals)
            for val in param_vals:
                input_map = {k: v for k, v in state['input_map'].items()}
                input_map[state['next_template_node']] = len(state['nodes'])
                cur_next_node = {
                    'function': next_node['function'],
                    'inputs': [input_map[idx] for idx in next_node['inputs']],
                    'value_inputs': [val],
                }
                cur_next_vals = {k: v for k, v in state['vals'].items()}
                if param_type == 'Shape':
                    if metadata['dataset'] in ['SHOP-VRB', 'SHOP-VRB-text']:
                        val = val + ' thing'
                        pass
                cur_next_vals[param_name] = val
                states.append({
                    'nodes': state['nodes'] + [cur_next_node],
                    'vals': cur_next_vals,
                    'input_map': input_map,
                    'next_template_node': state['next_template_node'] + 1,
                })
        else:
            input_map = {k: v for k, v in state['input_map'].items()}
            input_map[state['next_template_node']] = len(state['nodes'])
            next_node = {
                'function': next_node['function'],
                'inputs': [input_map[idx] for idx in next_node['inputs']],
                'value_inputs': []
            }
            states.append({
                'nodes': state['nodes'] + [next_node],
                'vals': state['vals'],
                'input_map': input_map,
                'next_template_node': state['next_template_node'] + 1,
            })

    # Actually instantiate the template with the solutions we've found
    text_questions, structured_questions, answers = [], [], []
    for state in final_states:
        structured_questions.append(state['nodes'])
        answers.append(state['answer'])
        text = random.choice(template['text'])
        for name, val in state['vals'].items():
            if val in synonyms:
                val = random.choice(synonyms[val])
            text = text.replace(name, str(val))
            text = ' '.join(text.split())
        text = adjust_spaces(text)
        text = adjust_plurals(text, plurals)
        text = replace_optionals(text)
        text = ' '.join(text.split())
        text = other_heuristic(text, state['vals'], metadata)
        text = replace_articles(text)
        text = replace_odd_phrasings(text, phrasings)
        text_questions.append(text)

    return text_questions, structured_questions, answers


def adjust_spaces(text):
    if 'thing s' in text:
        text = text.replace('thing s', 'things')
    if 'object s' in text:
        text = text.replace('object s', 'objects')
    if ' ;' in text:
        text = text.replace(' ;', ';')
    if ' ?' in text:
        text = text.replace(' ?', '?')
    return text


def replace_articles(text):
    splitted = text.split()
    for i in range(len(text.split()) - 1):
        if splitted[i] == 'a' and splitted[i + 1][0] in ['a', 'e', 'i', 'o', 'u']:
            splitted[i] = 'an'
    new_text = ' '.join(splitted)
    return new_text


def adjust_plurals(text, plurals):
    for word in text.split():
        word_adj = ''.join(x for x in word if x.isalpha())
        if word_adj in plurals:
            text = text.replace(word_adj, plurals[word_adj])
    return text


def replace_odd_phrasings(text, phrasings):
    # Dividided into simple text ones and regexes (no point for doing so many regexes)
    # Remember that text ones are executed first.
    for key in phrasings['text'].keys():
        if key in text:
            print("\n\n\nfound" + key + "\n\n\n")
            print(text)
    for key in phrasings['regex'].keys():
        text = re.sub(key, phrasings['regex'][key], text)
    return text


def replace_optionals(s):
    """
    Each substring of s that is surrounded in square brackets is treated as
    optional and is removed with probability 0.5. For example the string

    "A [aa] B [bb]"

    could become any of

    "A aa B bb"
    "A  B bb"
    "A aa B "
    "A  B "

    with probability 1/4.
    """
    pat = re.compile(r'\[([^\[]*)\]')

    while True:
        match = re.search(pat, s)
        if not match:
            break
        i0 = match.start()
        i1 = match.end()
        if random.random() > 0.5:
            s = s[:i0] + match.groups()[0] + s[i1:]
        else:
            s = s[:i0] + s[i1:]
    return s


def main(args):
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)
        dataset = metadata['dataset']
        if dataset not in ['SHOP-VRB', 'SHOP-VRB-text']:
            raise ValueError('Unrecognized dataset "%s"' % dataset)

    functions_by_name = {}
    for f in metadata['functions']:
        functions_by_name[f['name']] = f
    metadata['_functions_by_name'] = functions_by_name

    # Load templates from disk
    # Key is (filename, file_idx)
    num_loaded_templates = 0
    templates = {}
    q_individual_idx = dict()
    ind_idx = 0
    for fn in sorted(os.listdir(args.template_dir)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(args.template_dir, fn), 'r') as f:
            for i, template in enumerate(json.load(f)):
                num_loaded_templates += 1
                key = (fn, i)
                templates[key] = template
                q_individual_idx[key] = ind_idx
                ind_idx += 1
    print('Read %d templates from disk' % num_loaded_templates)

    def reset_counts():
        # Maps a template (filename, index) to the number of questions we have
        # so far using that template
        template_counts = {}
        # Maps a template (filename, index) to a dict mapping the answer to the
        # number of questions so far of that template type with that answer
        template_answer_counts = {}
        node_type_to_dtype = {n['name']: n['output'] for n in metadata['functions']}
        for key, template in templates.items():
            template_counts[key[:2]] = 0
            final_node_type = template['nodes'][-1]['function']
            final_dtype = node_type_to_dtype[final_node_type]
            answers = metadata['types'][final_dtype]
            if final_dtype == 'Bool':
                answers = [True, False]
            if final_dtype == 'Integer':
                if metadata['dataset'] in ['SHOP-VRB', 'SHOP-VRB-text']:
                    answers = list(range(0, 8))
            template_answer_counts[key[:2]] = {}
            for a in answers:
                template_answer_counts[key[:2]][a] = 0
        return template_counts, template_answer_counts

    template_counts, template_answer_counts = reset_counts()

    # Read file containing input scenes
    all_scenes = []
    with open(args.input_scene_file, 'r') as f:
        scene_data = json.load(f)
        all_scenes = scene_data['scenes']
        scene_info = scene_data['info']
    begin = args.scene_start_idx
    if args.num_scenes > 0:
        end = args.scene_start_idx + args.num_scenes
        all_scenes = all_scenes[begin:end]
    else:
        all_scenes = all_scenes[begin:]

    # Read synonyms file
    with open(args.synonyms_json, 'r') as f:
        synonyms = json.load(f)

    # Read plurals file
    with open(args.plurals_json, 'r') as f:
        plurals = json.load(f)

    # Read phrasings file
    with open(args.phrasings_json, 'r') as f:
        phrasings = json.load(f)

    questions = []
    q_len = 0
    scene_count = 0
    for i, scene in enumerate(all_scenes):
        scene_fn = scene['image_filename']
        # Make deepcopy of a scene as we are modifying it and we do not want mem leaks
        scene_struct = copy.deepcopy(scene)
        print('starting image %s (%d / %d)' % (scene_fn, i + 1, len(all_scenes)))

        if scene_count % args.reset_counts_every == 0:
            print('resetting counts')
            template_counts, template_answer_counts = reset_counts()
        scene_count += 1

        # Order templates by the number of questions we have so far for those
        # templates. This is a simple heuristic to give a flat distribution over
        # templates.
        templates_items = list(templates.items())
        random.shuffle(templates_items)
        templates_items = sorted(templates_items,
                                 key=lambda x: template_counts[x[0][:2]])
        num_instantiated = 0
        for (fn, idx), template in templates_items:
            if args.verbose:
                print('trying template ', fn, idx)
            if args.time_dfs and args.verbose:
                tic = time.time()
            ts, qs, ans = instantiate_templates_dfs(
                scene_struct,
                template,
                metadata,
                template_answer_counts[(fn, idx)],
                synonyms,
                plurals,
                phrasings,
                max_instances=args.instances_per_template,
                verbose=False)
            if args.time_dfs and args.verbose:
                toc = time.time()
                print('Took ', toc - tic)
            image_index = int(os.path.splitext(scene_fn)[0].split('_')[-1])
            if 'split' in scene_info:
                split = scene_info['split']
            else:
                split = 'new'
            for t, q, a in zip(ts, qs, ans):
                questions.append({
                    'split': split,
                    'image_filename': scene_fn,
                    'image_index': image_index,
                    'image': os.path.splitext(scene_fn)[0],
                    'question': t,
                    'program': q,
                    'answer': a,
                    'template_filename': fn,
                    'question_in_family_index': idx,
                    'question_family_index': q_individual_idx[(fn, idx)],
                    'question_index': len(questions),
                })
            if len(ts) > 0:
                if args.verbose:
                    print('got one!')
                num_instantiated += 1
                template_counts[(fn, idx)] += 1
            elif args.verbose:
                print('did not get any =(')
            if num_instantiated >= args.templates_per_image:
                break
        # Scene struct can be deleted now
        del(scene_struct)
        # Intermediate save
        if args.dump_output_every > 0:
            if (i + 1) % args.dump_output_every == 0:
                base, ext = os.path.splitext(args.output_questions_file)
                nums = "_{:06d}_{:06d}".format(i + 1 - args.dump_output_every, i)
                filename = base + nums + ext
                print('Intermediate save: %s' % filename)
                with open(filename, 'w') as f:
                    json.dump({
                        'info': scene_info,
                        'questions': questions,
                    }, f)
                q_len += len(questions)
                questions = []
        else:
            q_len = len(questions)

    print("Genarated questions: {}".format(q_len))
    if args.dump_output_every == 0:
        with open(args.output_questions_file, 'w') as f:
            print('Writing output to %s' % args.output_questions_file)
            json.dump({
                'info': scene_info,
                'questions': questions,
            }, f)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.profile:
        import cProfile
        cProfile.run('main(args)')
    else:
        main(args)
