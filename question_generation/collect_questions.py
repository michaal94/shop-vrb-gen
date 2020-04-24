import argparse
import json
import os
from datetime import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='output/questions')
parser.add_argument('--output_file', default='output/SHOP_VRB_questions.json')
parser.add_argument('--version', default='1.0')
parser.add_argument('--date', default=dt.today().strftime("%d/%m/%Y"))
parser.add_argument('--license',
                    default='Creative Commons Attribution (CC-BY 4.0)')


def main(args):
    input_files = sorted(os.listdir(args.input_dir))
    questions = []
    split = None
    curr_len = 0
    for f_name in input_files:
        if not f_name.endswith('.json'):
            continue
        path = os.path.join(args.input_dir, f_name)
        with open(path, 'r') as f:
            q_struct = json.load(f)
        if split is not None:
            msg = 'Input directory contains scenes from multiple splits'
            assert q_struct['questions'][0]['split'] == split, msg
        else:
            split = q_struct['questions'][0]['split']

        file_questions = q_struct['questions']
        file_questions.sort(key=lambda x: x['question_index'])
        for q in file_questions:
            q['question_index'] += curr_len
            if type(q['answer']) == bool:
                if q['answer']:
                    q['answer'] = 'yes'
                else:
                    q['answer'] = 'no'
            q['answer'] = str(q['answer'])
            print('Question number: ', str(q['question_index']))

        questions += file_questions
        curr_len = len(questions)

    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': split,
            'license': args.license
        },
        'questions': questions
    }

    with open(args.output_file, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)





