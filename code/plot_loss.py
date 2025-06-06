import numpy as np
import matplotlib.pyplot as plt
import logging 

def get_kl(file_name = './ckpt/ec/ec_3type_KD.log'):

    start = 4

    with open(file_name, 'r') as f:
        lines = f.readlines()

    lines = lines[:-3]
    loss_kls = []
    i = 0  
    for line in lines:
        if i < start:
            i += 1
            continue
        else:
            if i % 2 == 0:
                line = line.rstrip()
                losses = line.split(',')
                for loss in losses:
                    name, value = loss.split(':')
                    if name == ' loss_kl':
                        loss_kls.append(float(value))
                        # print(float(value))
                i += 1
            else:
                i += 1
                continue
    return loss_kls 

def plot_curves():
    # ec_3type_KD = get_kl(file_name='ckpt/ec/ec_3type_KD.log')
    # ec_3type_no_KD = get_kl(file_name='ckpt/ec/ec_3type_no_KD.log')

    fold_3type_KD = get_kl(file_name='ckpt/fold/fold_3type_KD.log')
    fold_teacher_no_KD = get_kl(file_name='ckpt/fold/fold_teacher_no_KD.log')
    # print(loss_kls)
    plt.plot(fold_3type_KD, label='with KL loss BP', color='red')
    plt.plot(fold_teacher_no_KD, label='w/o KL loss BP', color='green')
    plt.xlabel('Epoch')  
    plt.ylabel('Loss KL')
    plt.legend()
    # plt.title('Loss KL Curve')
    plt.savefig("fold_KL_Losses.png")

def plot_one_curve():
    fold_teacher_KD = get_kl(file_name='ckpt/fold/fold_teacher_KD.log')
    # ec_3type_no_KD = get_kl(file_name='ckpt/ec/ec_3type_no_KD.log')

    # print(loss_kls)
    plt.plot(fold_teacher_KD)
    plt.xlabel('Epoch')  
    plt.ylabel('Loss KL')
    plt.legend()
    # plt.title('Loss KL Curve')
    plt.savefig("fold_teacher_KD.png")
plot_one_curve()