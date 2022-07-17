import torch
import keyboard
import cv2
import time

from options.test_options import TestOptions
from models.model_modules import get_gen_model_arch_dict, get_disc_model_arch_dict
import utils


# Initial image is always given for inference
WARMUP_STEPS = 1
FPS = 30


def run_simulator(test_opts):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

    # To avoid GPU RAM surge when loading a model checkpoint.
    # Reference: https://pytorch.org/docs/stable/generated/torch.load.html
    model_path = torch.load(test_opts.model_path, map_location='cpu')
    train_opts = model_path['opts']

    if train_opts.dataset_name == 'gta':
        # 0: left, 1: no action, 2: right
        resized_img_size = (320, 192)
        left = [1, 0, 0]
        forward = [0, 1, 0]
        right = [0, 0, 1]
    else:
        raise NotImplementedError()
    
    num_components = 2 if train_opts.memory_dim is not None else 1
    if train_opts.dataset_name == 'gta':
        num_action_spaces = 3
    
    gen_model_arch_dict = get_gen_model_arch_dict(train_opts.dataset_name, num_components)
    disc_model_arch_dict = get_disc_model_arch_dict(train_opts.dataset_name)
    img_size = [int(size) for size in train_opts.img_size.split('x')]

    gen, _ = utils.create_models(train_opts, use_gpu, num_action_spaces, img_size, 
                                 gen_model_arch_dict, device, disc_model_arch_dict)

    utils.load_my_state_dict(gen, model_path['netG'])

    # Set batch size as 1 for inference
    gen.batch_size = 1

    # Load an initial image
    img = cv2.imread(test_opts.init_img_path)
    img = utils.reverse_color_order(img)
    img = utils.make_channels_first_and_normalize_img(img)

    imgs = [torch.tensor([img], dtype=torch.float32).cuda()]
    print('imgs:', len(imgs), imgs[0].shape, imgs[0])
    # Start an initial action with forward
    acts = [torch.tensor(forward, dtype=torch.float32).cuda()]
    print('acts:', len(acts), acts[0].shape, acts[0])

    utils.set_grads(gen, False)
    gen.eval()

    gen_img = None

    # simulator loop
    while True:
        frame_start_time = time.time()
        act_label = ''
        if keyboard.is_pressed('e'):
            exit()

        elif keyboard.is_pressed('r') or gen_img is None:
            # Run warmup to get initial values
            # warmup is set to 0, so initial image is going to be used as input
            gen_img, warmup_h_c, M, alpha_prev, m_vec_prev, out_imgs, zs, alphas, fine_mask_list, map_list, \
               unmasked_base_imgs, alpha_losses = gen.run_warmup_phase(imgs, acts, WARMUP_STEPS)
            h, c = warmup_h_c
            img = gen_img[0].cpu().numpy()
            img = utils.adjust_img_to_render(img, resized_img_size)
            cv2.imshow(f'{train_opts.dataset_name} - inference', img)
            cv2.waitKey(1000)
            continue
        
        elif keyboard.is_pressed('a'):
            act = torch.tensor([left], dtype=torch.float32).cuda()
            hidden_action = -1

        elif keyboard.is_pressed('d'):
            act = torch.tensor([right], dtype=torch.float32).cuda()
            hidden_action = 1

        else:
            act = torch.tensor([forward], dtype=torch.float32).cuda()
            hidden_action = 0

        gen_img, h, c, z, M, alpha_prev, m_vec_prev, _, _, _, _ = gen.proceed_step(
                                                            gen_img, h, c, act, M, alpha_prev, m_vec_prev)

        img = gen_img[0].cpu().numpy()
        img = utils.adjust_img_to_render(img, resized_img_size)
        rectangle = img.copy()
        cv2.rectangle(rectangle, (0, 0), (150, 30), (0, 0, 0), -1)
        img = cv2.addWeighted(rectangle, 0.6, img, 0.4, 0)
        cv2.putText(img, "Action:", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if hidden_action == -1:
            color = (55, 155, 255)
            text = "LEFT"
        elif hidden_action == 1:
            text = "RIGHT"
            color = (55, 155, 255)
        elif hidden_action == 0:
            text = "FORWARD"
            color = (55, 255, 55)

        cv2.putText(img, text, (80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imshow(f'{train_opts.dataset_name} - inference', img)
        cv2.waitKey(1)

        wait = 1 / FPS - (time.time() - frame_start_time)
        if wait > 0:
            time.sleep(wait)
    

if __name__ == '__main__':
    opts = TestOptions().parse()
    run_simulator(opts)