"""
Generate shapeworld reference games
"""

from shapely.geometry import Point, box
from shapely import affinity
import numpy as np
from numpy import random
from PIL import Image
import aggdraw
from enum import Enum
from tqdm import tqdm
import os
import multiprocessing as mp
from collections import namedtuple

DIM = 64
X_MIN, X_MAX = (8, 48)
ONE_QUARTER = (X_MAX - X_MIN) // 3
X_MIN_34, X_MAX_34 = (X_MIN + ONE_QUARTER, X_MAX - ONE_QUARTER)
BUFFER = 10
SIZE_MIN, SIZE_MAX = (3, 8)

TWOFIVEFIVE = np.float32(255)

SHAPES = ['circle', 'square', 'rectangle', 'ellipse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'gray']
VOCAB = ['gray', 'shape', 'blue', 'square', 'circle', 'green', 'red', 'rectangle', 'yellow', 'ellipse', 'white']
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}


MAX_PLACEMENT_ATTEMPTS = 5


class ShapeSpec(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2


class ConfigProps(Enum):
    SHAPE_1_COLOR = 0
    SHAPE_1_SHAPE = 1
    SHAPE_2_COLOR = 2
    SHAPE_2_SHAPE = 3
    RELATION_DIR = 4


SHAPE_SPECS = list(ShapeSpec)


def rand_size():
    return random.randint(SIZE_MIN, SIZE_MAX)


def rand_size_2():
    """Slightly bigger."""
    return random.randint(SIZE_MIN + 2, SIZE_MAX + 2)


def rand_pos():
    return random.randint(X_MIN, X_MAX)


class Shape:
    def __init__(self,
                 x=None,
                 y=None,
                 relation=None,
                 relation_dir=None,
                 color=None):
        if color is None:
            self.color = random.choice(COLORS)
        else:
            self.color = color
        if x is not None or y is not None:
            assert x is not None and y is not None
            assert relation is None and relation_dir is None
            self.x = x
            self.y = y
        elif relation is None and relation_dir is None:
            self.x = rand_pos()
            self.y = rand_pos()
        else:
            # Generate on 3/4 of image according to relation dir
            if relation == 0:
                # x matters - y is totally random
                self.y = rand_pos()
                if relation_dir == 0:
                    # Place right 3/4 of screen, so second shape
                    # can be placed LEFT
                    self.x = random.randint(X_MIN_34, X_MAX)
                else:
                    # Place left 3/4
                    self.x = random.randint(X_MIN, X_MAX_34)
            else:
                # y matters - x is totally random
                self.x = rand_pos()
                if relation_dir == 0:
                    # Place top 3/4 of screen, so second shape can be placed
                    # BELOW
                    # NOTE: Remember coords for y travel in opp dir
                    self.y = random.randint(X_MIN, X_MAX_34)
                else:
                    self.y = random.randint(X_MIN_34, X_MAX)
        self.init_shape()

    def draw(self, image):
        image.draw.polygon(self.coords, PENS[self.color])

    def intersects(self, oth):
        return self.shape.intersects(oth.shape)


class Ellipse(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size()
        # Dy must be at least 1.6x dx, to remove ambiguity with circle
        bigger = int(self.dx * min_skew)
        if bigger >= SIZE_MAX:
            smaller = int(self.dx / min_skew)
            assert smaller > SIZE_MIN, ("{} {}".format(smaller, self.dx))
            self.dy = random.randint(SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape = Point(self.x, self.y).buffer(1)
        shape = affinity.scale(shape, self.dx, self.dy)
        shape = affinity.rotate(shape, random.randint(360))
        self.shape = shape

        #  self.coords = [int(x) for x in self.shape.bounds]
        self.coords = np.round(np.array(self.shape.boundary).astype(np.int))
        #  print(len(np.array(self.shape.convex_hull)))
        #  print(len(np.array(self.shape.convex_hull.boundary)))
        #  print(len(np.array(self.shape.exterior)))
        self.coords = np.unique(self.coords, axis=0).flatten()


class Circle(Ellipse):
    def init_shape(self):
        self.r = rand_size()
        self.shape = Point(self.x, self.y).buffer(self.r)
        self.coords = [int(x) for x in self.shape.bounds]

    def draw(self, image):
        image.draw.ellipse(self.coords, BRUSHES[self.color])


class Rectangle(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size_2()
        bigger = int(self.dx * min_skew)
        if bigger >= SIZE_MAX:
            smaller = int(self.dx / min_skew)
            self.dy = random.randint(SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape = box(self.x, self.y, self.x + self.dx, self.y + self.dy)
        # Rotation
        shape = affinity.rotate(shape, random.randint(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()

    def draw(self, image):
        image.draw.polygon(self.coords, BRUSHES[self.color], PENS[self.color])


class Square(Rectangle):
    def init_shape(self):
        self.size = rand_size_2()
        shape = box(self.x, self.y, self.x + self.size, self.y + self.size)
        # Rotation
        shape = affinity.rotate(shape, random.randint(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()


SHAPE_IMPLS = {
    'circle': Circle,
    'ellipse': Ellipse,
    'square': Square,
    'rectangle': Rectangle,
    # TODO: Triangle, semicircle
}


class I:
    def __init__(self):
        self.image = Image.new('RGB', (DIM, DIM))
        #  self.draw = ImageDraw.Draw(self.image)
        self.draw = aggdraw.Draw(self.image)

    def draw_shapes(self, shapes, flush=True):
        for shape in shapes:
            shape.draw(self)
        if flush:
            self.draw.flush()

    def show(self):
        self.image.show()
        #  self.image.resize((64, 64), Image.ANTIALIAS).show()

    def array(self):
        return np.array(self.image, dtype=np.uint8)

    def float_array(self):
        return np.divide(np.array(self.image), TWOFIVEFIVE)

    def save(self, path, filetype='PNG'):
        self.image.save(path, filetype)


def random_shape():
    return random.choice(SHAPES)


def random_color():
    return random.choice(COLORS)


def random_shape_from_spec(spec):
    color = None
    shape = None
    if spec == ShapeSpec.SHAPE:
        shape = random_shape()
    elif spec == ShapeSpec.COLOR:
        color = random_color()
    elif spec == ShapeSpec.BOTH:
        shape = random_shape()
        color = random_color()
    else:
        raise ValueError("Unknown spec {}".format(spec))
    return (color, shape)


SpatialConfig = namedtuple('SpatialConfig', ['shapes', 'distractors', 'relation', 'dir'])
SingleConfig = namedtuple('SingleConfig', ['shape', 'color'])


def random_config_single():
    shape_spec = ShapeSpec(random.randint(3))
    shape = random_shape_from_spec(shape_spec)
    return SingleConfig(*shape)


def random_config_spatial():
    # 0 -> only shape specified
    # 1 -> only color specified
    # 2 -> only both specified
    shape_1_spec = ShapeSpec(random.randint(3))
    shape_2_spec = ShapeSpec(random.randint(3))
    shape_1 = random_shape_from_spec(shape_1_spec)
    shape_2 = random_shape_from_spec(shape_2_spec)
    if shape_1 == shape_2:
        return random_config_spatial()
    relation = random.randint(2)
    relation_dir = random.randint(2)
    return SpatialConfig([shape_1, shape_2], None, relation, relation_dir)


def add_shape_from_spec(spec,
                        relation,
                        relation_dir,
                        shapes=None,
                        attempt=1):
    if attempt > MAX_PLACEMENT_ATTEMPTS:
        return None
    color, shape_ = spec
    if shape_ is None:
        shape_ = random_shape()
    shape = SHAPE_IMPLS[shape_](
        relation=relation, relation_dir=relation_dir, color=color)
    if shapes is not None:
        for oth in shapes:
            if shape.intersects(oth):
                return add_shape_from_spec(
                    spec,
                    relation,
                    relation_dir,
                    shapes=shapes,
                    attempt=attempt + 1)
        shapes.append(shape)
        return shape
    return shape


def add_shape_rel(spec, oth_shape, relation, relation_dir):
    """
    Add shape, obeying the relation/relation_dir w.r.t. oth shape
    """
    color, shape_ = spec
    if shape_ is None:
        shape_ = random_shape()
    if relation == 0:
        new_y = rand_pos()
        if relation_dir == 0:
            # Shape must be LEFT of oth shape
            new_x = random.randint(X_MIN, oth_shape.x - BUFFER)
        else:
            # Shape RIGHT of oth shape
            new_x = random.randint(oth_shape.x + BUFFER, X_MAX)
    else:
        new_x = rand_pos()
        if relation_dir == 0:
            # BELOW (remember y coords reversed)
            new_y = random.randint(oth_shape.y + BUFFER, X_MAX)
        else:
            # ABOVE
            new_y = random.randint(X_MIN, oth_shape.y - BUFFER)
    return SHAPE_IMPLS[shape_](x=new_x, y=new_y, color=color)


def new_color(existing_color):
    new_c = existing_color
    while new_c == existing_color:
        new_c = random.choice(COLORS)
    return new_c


def new_shape(existing_shape):
    new_s = existing_shape
    while new_s == existing_shape:
        new_s = random.choice(SHAPES)
    return new_s


def invalidate_spatial(config):
    # Invalidate by randomly choosing one property to change:
    ((shape_1_color, shape_1_shape),
     (shape_2_color,
      shape_2_shape)), extra_shape_specs, relation, relation_dir = config
    properties = []
    if shape_1_color is not None:
        properties.append(ConfigProps.SHAPE_1_COLOR)
    if shape_1_shape is not None:
        properties.append(ConfigProps.SHAPE_1_SHAPE)
    if shape_2_color is not None:
        properties.append(ConfigProps.SHAPE_2_COLOR)
    if shape_2_shape is not None:
        properties.append(ConfigProps.SHAPE_2_SHAPE)
    properties.append(ConfigProps.RELATION_DIR)
    # Randomly select property to invalidate
    # TODO: Support for invalidating multiple properties
    invalid_prop = random.choice(properties)

    if invalid_prop == ConfigProps.SHAPE_1_COLOR:
        return ((new_color(shape_1_color), shape_1_shape),
                (shape_2_color,
                 shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_1_SHAPE:
        return ((shape_1_color, new_shape(shape_1_shape)),
                (shape_2_color,
                 shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_2_COLOR:
        return ((shape_1_color, shape_1_shape),
                (new_color(shape_2_color),
                 shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_2_SHAPE:
        return ((shape_1_color, shape_1_shape), (shape_2_color,
                                                 new_shape(shape_2_shape))
                ), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.RELATION_DIR:
        return ((shape_1_color, shape_1_shape),
                (shape_2_color,
                 shape_2_shape)), extra_shape_specs, relation, 1 - relation_dir
    else:
        raise RuntimeError


def fmt_config(config):
    if isinstance(config, SingleConfig):
        return _fmt_config_single(config)
    elif isinstance(config, SpatialConfig):
        return _fmt_config_spatial(config)
    else:
        raise NotImplementedError(type(config))


def _fmt_config_single(config):
    color, shape = config
    shape_txt = 'shape'
    color_txt = ''
    if shape is not None:
        shape_txt = shape
    if color is not None:
        color_txt = color + ' '
    return '{}{}'.format(color_txt, shape_txt)


def _fmt_config_spatial(config):
    (s1, s2), extra, relation, relation_dir = config
    if relation == 0:
        if relation_dir == 0:
            rel_txt = 'left'
        else:
            rel_txt = 'right'
    else:
        if relation_dir == 0:
            rel_txt = 'below'
        else:
            rel_txt = 'above'
    if s1[0] is None:
        s1_0_txt = ''
    else:
        s1_0_txt = s1[0]
    if s1[1] is None:
        s1_1_txt = 'shape'
    else:
        s1_1_txt = s1[1]
    if s2[0] is None:
        s2_0_txt = ''
    else:
        s2_0_txt = s2[0]
    if s2[1] is None:
        s2_1_txt = 'shape'
    else:
        s2_1_txt = s2[1]
    return '{} {} {} {} {}'.format(s1_0_txt, s1_1_txt, rel_txt,
                                   s2_0_txt, s2_1_txt)


def generate_spatial(mp_args):
    """
    Generate a single image
    """
    random.seed()
    n_images, correct, i, data_type, context = mp_args
    # Get shapes and relations
    imgs = np.zeros((n_images, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((n_images, ), dtype=np.uint8)
    config = random_config_spatial()
    # Minimum of 2 correct worlds/2 distractors
    if data_type == 'concept':
        n_target = 2
        n_distract = 2
    else:
        n_target = 1
        n_distract = n_images  # Never run out of distractors
    idx_rand = list(range(n_images))
    # random.shuffle(idx_rand)
    for w_idx in idx_rand:
        if n_target > 0:
            label = 1
            n_target -= 1
        elif n_distract > 0:
            label = 0
            n_distract -= 1
        else:
            label = (random.random() < correct)
        new_config = config if label else invalidate_spatial(config)
        (ss1, ss2), extra_shape_specs, relation, relation_dir = new_config
        s2 = add_shape_from_spec(ss2, relation, relation_dir)

        attempts = 0
        while attempts < MAX_PLACEMENT_ATTEMPTS:
            # TODO: Support extra shapes
            s1 = add_shape_rel(ss1, s2, relation, relation_dir)
            if not s2.intersects(s1):
                break
        else:
            # Failed
            raise RuntimeError

        # Create image and draw shapes
        img = I()
        img.draw_shapes([s1, s2])
        imgs[w_idx] = img.array()
        labels[w_idx] = label
    return imgs, labels, config, i


def invalidate_single(config):
    color, shape_ = config
    if shape_ is not None and color is not None:
        # Sample random part to invalidate
        # Here, we can invalidate shape, or invalidate color, OR invalidate both
        part_to_invalidate = random.randint(3)
        if part_to_invalidate == 0:
            return (new_color(color), shape_)
        elif part_to_invalidate == 1:
            return (color, new_shape(shape_))
        elif part_to_invalidate == 2:
            return (new_color(color), new_shape(shape_))
        else:
            raise RuntimeError
    elif shape_ is not None:
        assert color is None
        return (None, new_shape(shape_))
    elif color is not None:
        assert shape_ is None
        return (new_color(color), None)
    else:
        raise RuntimeError


def generate_single(mp_args):
    random.seed()
    n_images, correct, i, data_type, context = mp_args
    imgs = np.zeros((n_images, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((n_images, ), dtype=np.uint8)
    config = random_config_single()
    if context != None:
        is_none = True
        while is_none:
            target_color, target_shape = config
            if target_color is None or target_shape is None:
                config = random_config_single()
            else:
                is_none = False
    if data_type == 'concept':
        n_target = 2
        n_distract = 2
    else:
        n_target = 1
        n_distract = n_images  # Never run out of distractors
    idx = list(range(n_images))
    shapes = []
    colors = []
    for w_idx in idx:
        if n_target > 0:
            label = 1
            n_target -= 1
        elif n_distract > 0:
            label = 0
            n_distract -= 1
        else:
            label = (random.random() < correct)
        new_config = config if label else invalidate_single(config)

        color_, shape_ = new_config
        if shape_ is None:
            shape_ = random_shape()
            
        if context != None:
            target_color, target_shape = config
            if label == 1:
                shape_ = target_shape
                color_ = target_color
            if label == 0:
                if context == 'shape':
                    shape_ = target_shape
                    same_color = True
                    while same_color:
                        color_ = random_color()
                        if color_ != target_color:
                            same_color = False
                if context == 'color':
                    color_ = target_color
                    same_shape = True
                    while same_shape:
                        shape_ = random_shape()
                        if shape_ != target_shape:
                            same_shape = False
                if context == 'both':
                    if w_idx == 1:
                        shape_ = target_shape
                        same_color = True
                        while same_color:
                            color_ = random_color()
                            if color_ != target_color:
                                same_color = False
                    elif w_idx == 2:
                        color_ = target_color
                        same_shape = True
                        while same_shape:
                            shape_ = random_shape()
                            if shape_ != target_shape:
                                same_shape = False
                if context == 'none':
                    same_color = True
                    while same_color:
                        color_ = random_color()
                        if color_ != target_color:
                            same_color = False
                    same_shape = True
                    while same_shape:
                        shape_ = random_shape()
                        if shape_ != target_shape:
                            same_shape = False
        else:
            # shape generalization - train
            """
            if shape_ == 'square':
                square = True
            else:
                square = False
            while square:
                shape_ = random_shape()
                if shape_ != 'square':
                    square = False"""
            # shape generalization - test
            """
            if label == 1:
                shape_ = 'square'"""
            
            # color generalization - train
            """
            if color_ == 'red':
                red = True
            else:
                red = False
            while red:
                color_ = random_color()
                if color_ != 'red':
                    red = False"""
            # color generalization - test
            """
            if label == 1:
                color_ = 'red'"""

            # combo generalization - train
            """
            if (color_ == 'red' and shape_ == 'circle') or (color_ == 'blue' and shape_ == 'square') or (color_ == 'green' and shape_ == 'rectangle') or (color_ == 'yellow' and shape_ == 'ellipse') or (color_ == 'white' and shape_ == 'circle') or (color_ == 'gray' and shape_ == 'square'):
                combo = True
            else:
                combo = False
            while combo:
                color_ = random_color()
                shape_ = random_shape()
                if not ((color_ == 'red' and shape_ == 'circle') or (color_ == 'blue' and shape_ == 'square') or (color_ == 'green' and shape_ == 'rectangle') or (color_ == 'yellow' and shape_ == 'ellipse') or (color_ == 'white' and shape_ == 'circle') or (color_ == 'gray' and shape_ == 'square')):
                    combo = False"""
            # combo generalization - test
            """
            if label == 1:
                combos = [('red','circle'),('blue','square'),('green','rectangle'),('yellow','ellipse'),('white','circle'),('gray','square')]
                combo = combos[np.random.randint(0,len(combos))]
                color_ = combo[0]
                shape_ = combo[1]"""
                        
        shapes.append(shape_)
        colors.append(color_)
        shape = SHAPE_IMPLS[shape_](color=color_)
        
        # Create image and draw shape
        img = I()
        img.draw_shapes([shape])
        imgs[w_idx] = img.array()
        labels[w_idx] = label
    
    if colors.count(colors[0])==1 and shapes.count(shapes[0])==1:
        if np.random.randint(0,2) == 0:
            config = SingleConfig(colors[0],None)
        else:
            config = SingleConfig(None,shapes[0])
    elif colors.count(colors[0])==1:
        config = SingleConfig(colors[0],None)
    elif shapes.count(shapes[0])==1:
        config = SingleConfig(None,shapes[0])
    else:
        config = SingleConfig(colors[0],shapes[0])
        
    return imgs, labels, config, i


def generate(n,
             n_images,
             correct,
             data_type='concept',
             img_func=generate_spatial,
             float_type=False,
             n_cpu=None,
             pool=None,
             do_mp=True,
             verbose=False,
             context=None):
    if not do_mp and pool is not None:
        raise ValueError("Can't specify pool if do_mp=True")
    if do_mp:
        pool_was_none = False
        if pool is None:
            pool_was_none = True
            if n_cpu is None:
                n_cpu = mp.cpu_count()
            pool = mp.Pool(n_cpu)

    if data_type == 'concept':
        if n_images == 4:
            print("Warning: n_images == 4, min targets/distractors both 2, no variance")
        else:
            assert n_images > 4, "Too few n_images"
    elif data_type == 'reference':
        assert n_images > 1, "Too few n_images"
    else:
        raise NotImplementedError("data_type = {}".format(data_type))

    all_imgs = np.zeros((n, n_images, 64, 64, 3), dtype=np.uint8)
    all_labels = np.zeros((n, n_images), dtype=np.uint8)
    configs = []

    mp_args = [(n_images, correct, i, data_type, context) for i in range(n)]
    if do_mp:
        gen_iter = pool.imap(img_func, mp_args)
    else:
        gen_iter = map(img_func, mp_args)
    if verbose:
        gen_iter = tqdm(gen_iter, total=n)
    
    for imgs, labels, config, i in gen_iter:
        all_imgs[i, ] = imgs
        all_labels[i, ] = labels
        configs.append(config)
    if do_mp and pool_was_none:  # Remember to close the pool
        pool.close()
        pool.join()

    if float_type:
        all_imgs = np.divide(all_imgs, TWOFIVEFIVE)
        all_labels = all_labels.astype(np.float32)
    langs = np.array([fmt_config(c) for c in configs], dtype=np.unicode)
    return {'imgs': all_imgs, 'labels': all_labels, 'langs': langs}


def save_images(img_dir, data):
    # Save to test directory
    for instance_idx, (instance, instance_labels, *rest) in enumerate(data):
        for world_idx, (world, label) in enumerate(
                zip(instance, instance_labels)):
            Image.fromarray(world).save(
                os.path.join(img_dir, '{}_{}.png'.format(instance_idx, world_idx)))

    index_fname = os.path.join(img_dir, 'index.html')
    with open(index_fname, 'w') as f:
        # Sorry for this code
        f.write('''
            <!DOCTYPE html>
            <html>
            <head>
            <title>Shapeworld Fast</title>
            <style>
            img {{
                padding: 10px;
            }}
            img.yes {{
                background-color: green;
            }}
            img.no {{
                background-color: red;
            }}
            </style>
            </head>
            <body>
            {}
            </body>
            </html>
            '''.format(''.join(
            '<h1>{}</h1><p>{}</p>'.format(
                ' '.join(lang), ''.join(
                    '<img src="{}_{}.png" class="{}">'.format(
                        instance_idx, world_idx, 'yes' if label else 'no')
                    for world_idx, (
                        world,
                        label) in enumerate(zip(instance, instance_labels))))
            for instance_idx, (
                instance, instance_labels,
                lang, *rest) in enumerate(data))))
    np.savez_compressed('test.npz', imgs=data.imgs, labels=data.labels)


IMG_FUNCS = {
    'single': generate_single,
    'spatial': generate_spatial,
}


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Fast ShapeWorld',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--n_examples', type=int, default=1000, help='Number of examples')
    parser.add_argument(
        '--n_images', type=int, default=3, help='Images per example')
    parser.add_argument(
        '--correct', type=float, default=0.5, help='Avg correct proportion of images (concept only)')
    parser.add_argument(
        '--no_mp', action='store_true', help='Don\'t use multiprocessing')
    parser.add_argument(
        '--data_type', choices=['concept', 'reference'], default='reference',
        help='What kind of data to generate')
    parser.add_argument(
        '--img_type', choices=list(IMG_FUNCS.keys()), default='single',
        help='What kind of images to generate')
    parser.add_argument(
        '--vis_dir', default=None, type=str,
        help='If specified, save sample visualization of data (100 images) to this folder'
    )
    parser.add_argument(
        '--out', default='../data/single/reference-1000-10.npz',
        help='Save dataset to this file'
    )

    args = parser.parse_args()

    data_dir = './data/single/reference-1000-'
    files = [data_dir+'0.npz', data_dir+'1.npz', data_dir+'2.npz', data_dir+'3.npz', data_dir+'4.npz',data_dir+'5.npz', data_dir+'6.npz', data_dir+'7.npz', data_dir+'8.npz', data_dir+'9.npz',data_dir+'10.npz', data_dir+'11.npz', data_dir+'12.npz', data_dir+'13.npz', data_dir+'14.npz',data_dir+'15.npz', data_dir+'16.npz', data_dir+'17.npz', data_dir+'18.npz', data_dir+'19.npz',data_dir+'20.npz', data_dir+'21.npz', data_dir+'22.npz', data_dir+'23.npz', data_dir+'24.npz',data_dir+'25.npz', data_dir+'26.npz', data_dir+'27.npz', data_dir+'28.npz', data_dir+'29.npz',data_dir+'30.npz', data_dir+'31.npz', data_dir+'32.npz', data_dir+'33.npz', data_dir+'34.npz',data_dir+'35.npz', data_dir+'36.npz', data_dir+'37.npz', data_dir+'38.npz', data_dir+'39.npz',data_dir+'40.npz', data_dir+'41.npz', data_dir+'42.npz', data_dir+'43.npz', data_dir+'44.npz',data_dir+'45.npz', data_dir+'46.npz', data_dir+'47.npz', data_dir+'48.npz', data_dir+'49.npz',data_dir+'50.npz', data_dir+'51.npz', data_dir+'52.npz', data_dir+'53.npz', data_dir+'54.npz',data_dir+'70.npz', data_dir+'71.npz', data_dir+'72.npz', data_dir+'73.npz', data_dir+'74.npz']
    for file in files:
        data = generate(
            args.n_examples, args.n_images, args.correct, verbose=True,
            data_type=args.data_type,
            img_func=IMG_FUNCS[args.img_type],
            do_mp=not args.no_mp,
            context=None)
        np.savez_compressed(file, **data)
