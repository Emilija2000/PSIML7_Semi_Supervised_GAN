
import numpy as np
    
class twomoons_config:
    dataset = 'two_moons'
    num_label = 2
    data_noise = 0.1
    visualize = True
    draw_loss = True
    
    image_size = 1*2
    
    size_labeled_data = 1000
    size_unlabeled_data = 10000 
    size_dev = 1000

    gen_emb_size = 4
    noise_size = 100

    dis_lr = 9e-5
    enc_lr = 5e-5
    gen_lr = 5e-5

    eval_period = 50
    vis_period = 50
    loss_graph_period = 500
    gen_samples_period = 500

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 50000
    vi_weight = 1e-8

