import argparse
import json

"""
Used to append textual properties based on the predefined file
to the previously generated scenes
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_properties', default='./textual_properties_shop_vrb.json',
                    help="File with textual properties dictionary")
parser.add_argument('--input_scenes', default='../output/SHOP_VRB_collected_scenes.json')
parser.add_argument('--output_scenes', default='../output/SHOP_VRB_collected_scenes_text.json')


def main(args):
    with open(args.input_scenes, 'r') as f:
        scenes = json.load(f)

    with open(args.input_properties) as f:
        properties = json.load(f)

    scenes['scenes'] = scenes['scenes'][0:100]

    for scene in scenes['scenes']:
        for obj in scene['objects']:
            name = obj['name']
            if name not in properties:
                continue
            properties_set = properties[name]
            for key, prop in properties_set.items():
                if isinstance(prop, dict):
                    obj[key] = None
                    for criterion in prop.keys():
                        crit = criterion.split(":")
                        if len(crit) > 1:
                            if obj[crit[0]] == crit[1]:
                                obj[key] = prop[criterion]
                    if obj[key] is None:
                        obj[key] = prop['default']
                else:
                    obj[key] = prop

    with open(args.output_scenes, 'w') as f:
        json.dump(scenes, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
