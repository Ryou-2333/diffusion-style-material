import torch 
import numpy as np

eps = 1e-6

# set up size, light pos, light intensity
def set_param(device='cuda'):
	size = 4.0
	light_pos = torch.tensor([0.0, 0.0, 4], dtype=torch.float32).view(1, 3, 1, 1)
	light = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1) * 16 * np.pi
	light_pos = light_pos.to(device)
	light = light.to(device)
	return light, light_pos, size

def AdotB(a, b):
	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)

def norm(vec): #[B,C,W,H]
	vec = vec.div(vec.norm(2.0, 1, keepdim=True)+eps)
	return vec

def GGX(cos_h, alpha):
	c2 = cos_h**2
	a2 = alpha**2
	den = c2 * a2 + (1 - c2)
	return a2 / (np.pi * den**2 + 1e-6)

def Beckmann( cos_h, alpha):
	c2 = cos_h ** 2
	t2 = (1 - c2) / c2
	a2 = alpha ** 2
	return torch.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

def Fresnel(cos, f0):
	return f0 + (1 - f0) * (1 - cos)**5

def Fresnel_S(cos, specular):
	sphg = torch.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
	return specular + (1.0 - specular) * sphg

def Smith(n_dot_v, n_dot_l, alpha):
	def _G1(cos, k):
		return cos / (cos * (1.0 - k) + k)
	k = (alpha * 0.5).clamp(min=1e-6)
	return _G1(n_dot_v, k) * _G1(n_dot_l, k)

# def norm(vec): #[B,C,W,H]
# 	vec = vec.div(vec.norm(2.0, 1, keepdim=True))
# 	return vec

def getDir(pos, tex_pos):
	vec = pos - tex_pos
	return norm(vec), (vec**2).sum(1, keepdim=True)

# def AdotB(a, b):
# 	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
def getTexPos(res, size, device='cpu'):
	x = torch.arange(res, dtype=torch.float32)
	x = ((x + 0.5) / res - 0.5) * size

	# surface positions,
	y, x = torch.meshgrid((x, x))
	z = torch.zeros_like(x)
	pos = torch.stack((x, -y, z), 0).to(device)

	return pos

# point light
def render(maps, tex_pos, li_color, li_pos, post='gamma', device='cuda', isMetallic=False, no_decay=False, amb_li=False, cam_pos=None, dir_flag=False):

	if len(li_pos.shape)!=4:
		li_pos = li_pos.unsqueeze(-1).unsqueeze(-1)

	assert len(li_color.shape)==4, "dim of the shape of li_color pos should be 4"
	assert len(li_pos.shape)==4, f"dim of the shape of camlight pos {li_pos.shape} should be 4"
	assert len(tex_pos.shape)==4, "dim of the shape of position map should be 4"
	assert len(maps.shape)==4, "dim of the shape of feature map should be 4"
	assert li_pos.shape[1]==3, "the 1 channel of position map should be 3"

	# print(" maps: ",maps.shape)
	if maps.shape[1]==12:
		use_spec = True
		spec = maps[:,9:12,:,:]
	else:
		use_spec = False

	# print('use_spec: ', use_spec)

	if cam_pos is None:
		cam_pos = li_pos

	normal = maps[:,0:3,:,:]
	albedo = maps[:,3:6,:,:]
	rough = maps[:,6:9,:,:]
	if isMetallic:
		metallic = maps[:,9:12,:,:]
		f0 = 0.04
		# update albedo using metallic
		f0 = f0 + metallic * (albedo - f0)
		albedo = albedo * (1.0 - metallic) 
	else:
		f0 = 0.04

	if dir_flag:
		l = norm(li_pos)

		# v_dir = torch.zeros_like(l).to(l.device)
		# v_dir[:,-1,:,:]=1
		# cos = compute_cos(l, v_dir)
		# dist_l_sq = (4/cos)**2

		dist_l_sq = 16

		v = l
	else:
		l, dist_l_sq = getDir(li_pos, tex_pos)
		v, _ = getDir(cam_pos, tex_pos)


	h = norm(l + v)
	normal = norm(normal)

	n_dot_v = AdotB(normal, v)
	n_dot_l = AdotB(normal, l)
	n_dot_h = AdotB(normal, h)
	v_dot_h = AdotB(v, h)

	# print('dist_l_sq:',dist_l_sq)
	if no_decay:
		geom = n_dot_l
	else:
		geom = n_dot_l / (dist_l_sq + eps)

	D = GGX(n_dot_h, rough**2)
	if use_spec:
		F = Fresnel_S(v_dot_h, spec)
	else:
		F = Fresnel(v_dot_h, f0)
	G = Smith(n_dot_v, n_dot_l, rough**2)

	## lambert brdf
	f1 = albedo / np.pi
	if use_spec:
		f1 *= (1-spec)

	## cook-torrance brdf
	f2 = D * F * G / (4 * n_dot_v * n_dot_l + eps)
	f = f1 + f2
	img = f * geom * li_color

	if amb_li:
		amb_intensity = torch.rand(1).item() * 0.1
		amb_light = torch.rand([img.shape[0],img.shape[1],1,1], device=device)*amb_intensity
		# amb_light = albedo*amb_intensity

		img = img + amb_light

	if post=='gamma':
		# print('gamma')
		return img.clamp(eps, 1.0)**(1/2.2)
	elif post=='reinhard':
		# print('reinhard')
		img = img.clamp(min=eps)
		return img/(img+1)
	elif post=='hdr':
		# print('hdr')
		return img.clamp(min=eps)

#[B,c,H,W]
def height_to_normal(img_in, size, intensity=0.2): # 0.02 for debugging, 0.2 is regular setting

	"""Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

	Args:
		img_in (tensor): Input image.
		mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
		normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
		use_input_alpha (bool, optional): Use input alpha. Defaults to False.
		use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
		intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
		max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

	Returns:
		Tensor: Normal image.
	"""
	# grayscale_input_check(img_in, "input height field")
	assert img_in.shape[1]==1, 'should be grayscale image'

	def roll_row(img_in, n):
		return img_in.roll(n, 2)

	def roll_col(img_in, n):
		return img_in.roll(n, 3)

	def norm(vec): #[B,C,W,H]
		vec = vec.div(vec.norm(2.0, 1, keepdim=True))
		return vec
	
	img_in = img_in*intensity

	dx = (roll_col(img_in, 1) - roll_col(img_in, -1))
	dy = (roll_row(img_in, 1) - roll_row(img_in, -1))
	
	pixSize = size / img_in.shape[-1]
	dx /= 2 * pixSize
	dy /= 2 * pixSize

	img_out = torch.cat((dx, -dy, torch.ones_like(dx)), 1)
	img_out = norm(img_out)
	# img_out = img_out / 2.0 + 0.5 #[-1,1]->[0,1]
	
	return img_out
