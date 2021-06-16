# SHOP-VRB Dataset Generation

The code was used to generate the [SHOP-VRB dataset](https://michaal94.github.io/SHOP-VRB/) as described in the paper:

**[SHOP-VRB: A Visual Reasoning Benchmark for Object Perception](https://arxiv.org/abs/2004.02673)**
 <br>
 Michal Nazarczuk, Krystian Mikolajczyk
 <br>
 Accepted at ICRA 2020

The code can be used to generate synthetic images and corresponding questions concerning household items typically seen on the kitchen tabletop.

The code was developed based on [CLEVR generation code](https://github.com/facebookresearch/clevr-dataset-gen).

## Generating Images

Firstly, we use [Blender](https://www.blender.org/) to render synthetic images of the scenes along with JSON file with ground-truth information about the scene.

We suggest using Blender in version at least 2.8x as it comes with denoising filters that allow to render glossy surfaces with much lower cost and without *firefly* artifacts. Additionally, newer Blender version support rendering with OptiX - Nvidia engine for RTX cards (slightly faster than rendering with CUDA).

Blender comes with Python installation built-in. However, we need to add some modules (i.e. *pycocotools*). You can add it directly to Blender Python installation or link other Python installation to Blender (you can symlink your Python to the Blender directory), e.g:

```bash
ln -s /home/uname/anaconda3/envs/blender_python /home/uname/.local/share/blender/2.81/python
```

Remember that when using the latter way, you need to match Python version to the one bundled with your Blender version.

Then, you need to add the `image_generation` directory to Python path of Blender's Python path. You can add a `.pth` file to the `site-packages` directory of Blender's Python, like this:

```bash
echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/python3.x/site-packages/shop_vrb.pth
```

where `$BLENDER` is the directory where Blender is installed and `$VERSION` is your Blender version.

You can then render some images like this:

```bash
cd image_generation
blender --background --python render_images.py -- --num_images 10
```

where `blender` may be an alias leading to your Blender app.

If you have an NVIDIA GPU with CUDA installed then you can use the GPU to accelerate rendering like this:

```bash
blender --background --python render_images.py -- --num_images 10 --use_gpu
```

By default rendered images are stored in `output/images` and ground-truth scene information is placed in `output/SHOP_VRB_scenes.json`.

You can find [more details about image rendering here](image_generation/README.md).

## Generating Questions
Further, we generate questions, programs, and answers for the previously generated scenes. Based on the input ground truth scene, a JSON containing questions, programs and answers is generated.

You can generate questions like this:

```bash
cd question_generation
python generate_questions.py
```

The file `output/SHOP_VRB_questions.json` will then contain questions for the generated images.

You can [find more details about question generation here](question_generation/README.md).

## Citation:
```
@article{nazarczuk2020shop,
  title={SHOP-VRB: A Visual Reasoning Benchmark for Object Perception},
  author={Nazarczuk, Michal and Mikolajczyk, Krystian},
  journal={International Conference on Robotics and Automation (ICRA)},
  year={2020}
  }
```
