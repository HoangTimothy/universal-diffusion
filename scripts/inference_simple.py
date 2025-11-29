"""
Script đơn giản để inference/sampling với pretrained model.
Có thể chạy trên CPU/GPU (không bắt buộc TPU) nếu sửa code.

Usage:
  python3 scripts/inference_simple.py \
    --model_dir /path/to/checkpoint \
    --checkpoint_name model.ckpt-1000000 \
    --dataset cifar10 \
    --num_samples 100 \
    --output_dir ./samples
"""

import argparse
import os
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image

# Import từ codebase
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_tf import utils
from diffusion_tf.diffusion_utils_2 import get_beta_schedule, GaussianDiffusion2
from diffusion_tf.models import unet
from diffusion_tf.tpu_utils import tpu_utils, datasets


def load_model_and_sample(model_dir, checkpoint_name, dataset_name, num_samples, output_dir, 
                          use_tpu=False, tpu_name=None, bucket_name_prefix=None):
    """
    Load model và tạo samples.
    
    Args:
        model_dir: Đường dẫn đến thư mục checkpoint
        checkpoint_name: Tên checkpoint (ví dụ: "model.ckpt-1000000")
        dataset_name: Tên dataset ("cifar10", "celebahq256", "lsun")
        num_samples: Số samples cần tạo
        output_dir: Thư mục lưu samples
        use_tpu: Có dùng TPU không
        tpu_name: Tên TPU (nếu dùng)
        bucket_name_prefix: Prefix GCS bucket (nếu dùng)
    """
    
    # Load config từ checkpoint
    kwargs = tpu_utils.load_train_kwargs(model_dir)
    print('Loaded config:', kwargs)
    
    # Setup dataset để biết image shape
    if bucket_name_prefix:
        region = utils.get_gcp_region()
        tfds_data_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, 'tensorflow_datasets')
    else:
        tfds_data_dir = 'tensorflow_datasets'  # Local
    
    try:
        ds = datasets.get_dataset(dataset_name, tfds_data_dir=tfds_data_dir)
    except:
        # Fallback: dùng config từ kwargs
        print("Warning: Could not load dataset, using config from kwargs")
        img_size = {'cifar10': 32, 'celebahq256': 256}.get(dataset_name, 32)
        ds = type('obj', (object,), {
            'num_classes': 1,
            'image_shape': [img_size, img_size, 3]
        })()
    
    # Tạo model
    betas = get_beta_schedule(
        kwargs['beta_schedule'], 
        beta_start=kwargs['beta_start'], 
        beta_end=kwargs['beta_end'],
        num_diffusion_timesteps=kwargs['num_diffusion_timesteps']
    )
    
    # Model class (giống trong run_cifar.py)
    class Model(tpu_utils.Model):
        def __init__(self):
            self.diffusion = GaussianDiffusion2(
                betas=betas, 
                model_mean_type=kwargs['model_mean_type'],
                model_var_type=kwargs['model_var_type'], 
                loss_type=kwargs['loss_type']
            )
            self.num_classes = ds.num_classes
        
        def _denoise(self, x, t, y, dropout):
            B, H, W, C = x.shape.as_list()
            assert x.dtype == tf.float32
            assert t.shape == [B] and t.dtype in [tf.int32, tf.int64]
            out_ch = (C * 2) if self.diffusion.model_var_type == 'learned' else C
            y = None
            
            if kwargs['model_name'] == 'unet2d16b2':
                return unet.model(
                    x, t=t, y=y, name='model', ch=128, ch_mult=(1, 2, 2, 2), 
                    num_res_blocks=2, attn_resolutions=(16,),
                    out_ch=out_ch, num_classes=self.num_classes, dropout=dropout
                )
            elif kwargs['model_name'] == 'unet2d16b2c112244':
                return unet.model(
                    x, t=t, y=y, name='model', ch=128, ch_mult=(1, 1, 2, 2, 4, 4), 
                    num_res_blocks=2, attn_resolutions=(16,),
                    out_ch=out_ch, num_classes=self.num_classes, dropout=dropout
                )
            else:
                raise NotImplementedError(kwargs['model_name'])
        
        def samples_fn(self, dummy_noise, y):
            import functools
            return {
                'samples': self.diffusion.p_sample_loop(
                    denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
                    shape=dummy_noise.shape.as_list(),
                    noise_fn=tf.random_normal
                )
            }
    
    # Setup TPU hoặc CPU/GPU
    if use_tpu and tpu_name:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        master = resolver.master()
    else:
        strategy = None
        master = ''
    
    # Tạo graph
    with tf.Graph().as_default():
        if strategy:
            with strategy.scope():
                model = Model()
        else:
            model = Model()
        
        # Tạo samples
        img_shape = ds.image_shape
        batch_size = min(16, num_samples)  # Batch size nhỏ cho CPU/GPU
        
        dummy_noise = tf.placeholder(tf.float32, shape=[batch_size] + img_shape)
        y = tf.placeholder(tf.int32, shape=[batch_size])
        samples = model.samples_fn(dummy_noise, y)['samples']
        
        # Load checkpoint
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(model_dir, checkpoint_name)
        
        # Tạo output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Session và sampling
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        
        with tf.Session(target=master, config=config) as sess:
            print('Loading checkpoint:', checkpoint_path)
            saver.restore(sess, checkpoint_path)
            print('Checkpoint loaded!')
            
            # Generate samples
            all_samples = []
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                print(f'Generating batch {i+1}/{num_batches}...')
                batch_samples = sess.run(
                    samples,
                    feed_dict={
                        dummy_noise: np.random.randn(batch_size, *img_shape).astype(np.float32),
                        y: np.zeros(batch_size, dtype=np.int32)
                    }
                )
                all_samples.append(batch_samples)
            
            # Concatenate và unnormalize
            all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
            
            # Unnormalize: [-1, 1] -> [0, 255]
            unnormalize = lambda x: np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8)
            all_samples = unnormalize(all_samples)
            
            # Lưu samples
            print(f'Saving {num_samples} samples to {output_dir}...')
            for i, sample in enumerate(all_samples):
                img = Image.fromarray(sample)
                img.save(os.path.join(output_dir, f'sample_{i:05d}.png'))
            
            # Lưu numpy array
            np.save(os.path.join(output_dir, 'samples.npy'), all_samples)
            print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple inference script for DDPM')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='Checkpoint name (e.g., model.ckpt-1000000)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'celebahq256', 'lsun'],
                        help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./samples',
                        help='Output directory for samples')
    parser.add_argument('--use_tpu', action='store_true',
                        help='Use TPU (requires tpu_name)')
    parser.add_argument('--tpu_name', type=str, default=None,
                        help='TPU name')
    parser.add_argument('--bucket_name_prefix', type=str, default=None,
                        help='GCS bucket prefix (if using GCS)')
    
    args = parser.parse_args()
    
    load_model_and_sample(
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_tpu=args.use_tpu,
        tpu_name=args.tpu_name,
        bucket_name_prefix=args.bucket_name_prefix
    )


