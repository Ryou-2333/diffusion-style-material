import torch
from utils import instantiate_from_config
from stylegan_interface import load_generator_decoder, generate_carpaint
import yaml
from PIL import Image
from torchvision import transforms

MODEL_DICTS = {
    "cls-mlp": "configs/mlp.yaml",
}

CKPT_DICTS = {
    "cls-mlp": "checkpoints/cls-mlp/latest.ckpt",
}

gen_path = 'weights/photomat/G_512.pkl'
dec_path='weights/photomat/MatUnet_512.pt'

transform = transforms.Compose([
    transforms.ToTensor()
])

def load_models(model_name):
    config_file = MODEL_DICTS[model_name]
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f.read())
    model = instantiate_from_config(configs["model"]).eval()
    model.init_from_ckpt(CKPT_DICTS[model_name])
    gen, dec, res = load_generator_decoder(gen_path, dec_path, device=model.device)
    return model, gen, dec, res

def test_model(model_name, input_pth, output_dir, output_inter, count=1):
    img = Image.open(input_pth)
    model, gen, dec, res = load_models(model_name)
    img_t = transform(img).unsqueeze(0).to(model.device)
    for i in range(count):
        w, inters = model.generate_w(img_t, 1, seed=i * 23572)
        w_s = w.repeat([1, 16, 1]).to(model.device)
        out = generate_carpaint(gen, dec, w_s, res, device=model.device)
        out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        out = out.squeeze(0).detach()
        out = Image.fromarray(out.cpu().numpy(), 'RGB')
        out.save(f"{output_dir}/result_{i}.png")
        if(output_inter):
            j = 0
            for inter in inters:
                i_s = inter.repeat([1, 16, 1]).to(model.device)
                out = generate_carpaint(gen, dec, i_s, res, device=model.device)
                out = (out*255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                out = out.squeeze(0).detach()
                out = Image.fromarray(out.cpu().numpy(), 'RGB')
                out.save(f"{output_dir}/inter_{i}_{j}.png")
                j += 1

test_model("cls-mlp", "checkpoints/cls-mlp/test/wall-1.png", "checkpoints/cls-mlp/test", False, 4)

