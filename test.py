import torch
from utils import instantiate_from_config
from stylegan_interface import load_generator_decoder, generate_carpaint
import yaml
from PIL import Image
from torchvision import transforms
import numpy as np
import random

MODEL_DICTS = {
    "cls-mlp": "configs/mlp.yaml",
    "local-attn": "configs/local.yaml",
    "full-dp10": "configs/local.yaml"
}

CKPT_DICTS = {
    "cls-mlp": "checkpoints/cls-mlp/latest.ckpt",
    "local-attn": "checkpoints/local-dir/epoch-10.ckpt",
    "full-dp10": "checkpoints/full-dir-dp15/epoch-25.ckpt"
}

gen_path = 'weights/photomat/G_512.pkl'
dec_path='weights/photomat/MatUnet_512.pt'

transform = transforms.Compose([
    transforms.ToTensor()
])

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

def load_models(model_name):
    config_file = MODEL_DICTS[model_name]
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f.read())
    model = instantiate_from_config(configs["model"]).eval().cuda()
    model.init_from_ckpt(CKPT_DICTS[model_name])
    gen, dec, res = load_generator_decoder(gen_path, dec_path, device=model.device)
    return model, gen, dec, res

def test_model(model_name, input_pth, output_dir, output_inter, count=1, gs = 1.0, random_seed = False, name=0):
    img = Image.open(input_pth)
    model, gen, dec, res = load_models(model_name)
    img_t = transform(img).unsqueeze(0).to(model.device)
    for i in range(count):
        if random_seed: 
            seed=random.randint(min_seed_value, max_seed_value)
        else:
            seed=i * 23572
        w, inters = model.generate_w(img_t, 1, unconditional_guidance_scale=gs, seed=seed)
        w_s = w.repeat([1, 16, 1]).to(model.device)
        out = generate_carpaint(gen, dec, w_s, res, device=model.device)
        out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        out = out.squeeze(0).detach()
        out = Image.fromarray(out.cpu().numpy(), 'RGB')
        out.save(f"{output_dir}/result_{name}_gs{gs}_{seed}.png")
        if(output_inter):
            j = 0
            for inter in inters:
                i_s = inter.repeat([1, 16, 1]).to(model.device)
                out = generate_carpaint(gen, dec, i_s, res, device=model.device)
                out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                out = out.squeeze(0).detach()
                out = Image.fromarray(out.cpu().numpy(), 'RGB')
                out.save(f"{output_dir}/inter_{name}_gs{gs}_{j}_{seed}.png")
                j += 1

#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.1)
#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.2)
#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.3)
#test_model("full-dp10", "checkpoints/full-dir-dp10/4_0_sampled.png", "checkpoints/full-dir-dp10", False, 3, 1.4)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 1, name=0)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 2, name=0)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 3, name=0)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 5, name=0)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 6, name=0)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 7, name=0)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 8, name=0)
test_model("full-dp10", "checkpoints/full-dir-dp15/plank01_rendered.png", "checkpoints/full-dir-dp15", False, 12, 9, name=0)