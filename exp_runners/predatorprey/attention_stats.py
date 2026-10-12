import numpy as np
import collections
import tikzplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from matplotlib import rcParams
import pandas as pd
from collections import defaultdict
import os

import math
import joblib

rcParams['mathtext.fontset'] = 'cm'
rcParams['figure.figsize'] = [4, 4]

def plot_attn_stats(distance_vs_weight, exp_dir, penalty):
    fig, axs = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.4)
    scatter_plt, avg_weight_plt, n_pts_plt = axs[0], axs[1], axs[2]
    distance_vs_avgweight = {}
    distance_vs_stdweight = {}
    distance_vs_n = {}
    for d in distance_vs_weight.keys():
        weights = distance_vs_weight[d]
        distance_vs_avgweight[d] = np.mean(weights)
        distance_vs_stdweight[d] = np.std(weights)
        distance_vs_n[d] = len(weights)
        scatter_plt.scatter([d] * len(weights), weights,
                    color='red', s=0.1,)
        scatter_plt.scatter(d, distance_vs_avgweight[d], 
                    color='black', marker='_')
    scatter_plt.set_xlabel('distance')
    scatter_plt.set_ylabel('attention weight')
    
    distance_vs_avgweight = \
        collections.OrderedDict(sorted(distance_vs_avgweight.items()))
    distance_vs_stdweight = \
        collections.OrderedDict(sorted(distance_vs_stdweight.items()))
    distance_vs_n = \
        collections.OrderedDict(sorted(distance_vs_n.items()))

    _x, _y, _y_err = [], [], []
    for d in distance_vs_avgweight.keys():
        _x.append(d)
        _y.append(distance_vs_avgweight[d])
        _y_err.append(distance_vs_stdweight[d])
    _x = np.array(_x)
    _y = np.array(_y)
    _y_err = np.array(_y_err)
    avg_weight_plt.plot(_x, _y, color='blue')
    avg_weight_plt.fill_between(_x, 
                                _y - _y_err / 2, 
                                _y + _y_err / 2,
                                color='blue', 
                                alpha=0.2)
    avg_weight_plt.set_xlabel('Distance')
    avg_weight_plt.set_ylabel('Avg. Attention Weight')
    # avg_weight_plt.set_title('predatorprey, grid={}, n_agents={}, n_preys={}'.
    #     format(args.grid_size, args.n_agents, args.n_preys))

    _x, _y = [], []
    for d in distance_vs_n.keys():
        _x.append(d)
        _y.append(distance_vs_n[d])
    _y = np.array(_y[1:]) / np.sum(_y[1:])
    n_pts_plt.plot(_x[1:], _y) # discard self attention
    n_pts_plt.set_xlabel('Distance')
    n_pts_plt.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig('./plots/penalty={}_AttnWeights.png'.format(penalty))

def plot_attn_stats_simple(distance_vs_weight, results_folder, penalty, n_samples, de,attention_type, settings,x_type, y_type):
    fig, axs = plt.subplots()
    avg_weight_plt = axs
    distance_vs_avgweight = {}
    distance_vs_stdweight = {}
    distance_vs_n = {}
    for d in distance_vs_weight.keys():
        weights = distance_vs_weight[d]
        distance_vs_avgweight[d] = np.mean(weights)
        distance_vs_stdweight[d] = np.std(weights)
        distance_vs_n[d] = len(weights)

    distance_vs_avgweight = \
        collections.OrderedDict(sorted(distance_vs_avgweight.items()))
    distance_vs_stdweight = \
        collections.OrderedDict(sorted(distance_vs_stdweight.items()))
    distance_vs_n = \
        collections.OrderedDict(sorted(distance_vs_n.items()))

    _x, _y, _y_err, _n = [], [], [], []
    for d in distance_vs_avgweight.keys():
        _x.append(d)
        _y.append(distance_vs_avgweight[d])
        _y_err.append(distance_vs_stdweight[d])
        _n.append(distance_vs_n[d])
    _x = np.array(_x)
    _y = np.array(_y)
    _y_err = np.array(_y_err)
    _n = np.array(_n)
    avg_weight_plt.plot(_x, _y, color='blue')

    avg_weight_plt.fill_between(_x,
                                _y - _y_err,
                                _y + _y_err,
                                color='blue',
                                alpha=0.2)
    
    avg_weight_plt.set_xlabel(x_type, fontsize=16)
    avg_weight_plt.set_ylabel(y_type, fontsize=16)

    case_study_dir = os.path.join(results_folder, "case_study/")
    os.makedirs(case_study_dir, exist_ok=True)

    fig.tight_layout()
    if de:
        results_name = case_study_dir+'m={}_penalty={}_{}_DE_AttnWeights_{}_{}'.format(attention_type, penalty,settings,x_type,y_type)
        fig.savefig(results_name+".pdf")
    else:
        results_name = case_study_dir+'m={}_penalty={}_{}_CE_AttnWeights_{}_{}'.format(attention_type, penalty,settings,x_type,y_type)
        fig.savefig(results_name+".pdf")

    distance_vs_avgweight_df = pd.DataFrame(distance_vs_avgweight, index=[0])
    distance_vs_stdweight_df = pd.DataFrame(distance_vs_stdweight, index=[0])
    results = pd.concat([distance_vs_avgweight_df, distance_vs_stdweight_df])
    results.index = ['average', 'std']
    print(results)

    outputpath = results_name+".csv"
    results.to_csv(outputpath, sep=',', index=True, header=True)

def find_folders_with_string(directory, raw_expression, seeds):
    matching_folders = []
    raw_expression = directory + raw_expression
    filenames = glob.glob(raw_expression)
    matching_folders.extend(filenames)

    return matching_folders

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def aggregate_dict_by_intervals(input_dict):
    # 创建区间到值的聚合字典
    # aggregated = defaultdict(list)
    dic = {}
    for key, value in input_dict.items():
        # 计算键所属的区间（0.0-0.1, 0.1-0.2, ..., 0.9-1.0）
        interval = int(key * 10) / 10 + 0.5
        # 处理边界情况：当 key=1.0 时
        if key == 1.0:
            interval = 0.9 + 0.5  # 归入最后一个区间 [0.9, 1.0]
        # 将值添加到对应区间
        if interval not in dic.keys():
            dic[interval] =[value]
        else:
            dic[interval].append(value)
        # aggregated[interval].append(value)

    return dic

def attn_analysis3(results_folder,exp_folder, args, seeds, de=False):
    distance_vs_weight = {}
    phy_d_vs_similarity = {}
    phy_d_vs_weight = {}
    normalize = True
    # normalize = False

    # remove_dia = True
    remove_dia = False

    # plotMatrix=True
    plotMatrix = False

    # print (results_folder)
    # print (exp_folder)
    # ("ce", "general", 1), ("de", "general", 1), ("ce", "polarized3", 2), ("de", "polarized3", 2)
    raw_expression = exp_folder+"_seed=*"
    matching_folders = find_folders_with_string(results_folder, raw_expression, seeds)
    print (len(matching_folders), matching_folders)

    for exp_dir in matching_folders:
        # print (exp_dir)
        seed = exp_dir[-1]
        # print (seed)
        data = joblib.load(exp_dir + '/params.pkl')
        env = data['env']
        algo = data['algo']
        n_agents = env.n_agents
        # n_agents = 1
        # args.n_eval_episodes = 2
        print("------", exp_dir)

        # args.n_eval_episodes = 5
        # args.max_env_steps = 20
        # args.n_eval_episodes = 300
        for i_eps in range(args.n_eval_episodes):
            print('Eval episode: {}/{}'.format(i_eps + 1, args.n_eval_episodes))
            # if i_eps>=3:
            #     continue
            obses = env.reset()
            algo.policy.reset([True])
            for i_step in range(args.max_env_steps):
                actions, agent_infos = algo.policy.get_actions(obses,
                                                               env.get_avail_actions(),
                                                               greedy=args.eval_greedy)
                if de:
                    attention_weights = algo.baseline.get_attention_weights(obses)
                    attention_matrix = attention_weights.detach().numpy()
                else:
                    # attention_weights_0 = agent_infos['attention_weights'][a]
                    attention_matrix = agent_infos['attention_weights']

                if normalize:
                    attention_matrix_array = np.array(attention_matrix)
                    if remove_dia:
                        np.fill_diagonal(attention_matrix_array, 0)

                    # print(attention_matrix_array.shape)
                    degree = np.sum(attention_matrix_array, axis=-1)
                    # degree_inv = np.diag(np.sqrt(degree))
                    # degree_inv = np.diag(degree**(-1/2))
                    # normalized_attention_weights = (degree_inv.dot(attention_matrix_array)).dot(degree_inv)

                    # degree_inv = np.diag(degree ** (-1))
                    degree_inv = np.diag(1.0 / (degree + 1e-12))
                    normalized_attention_weights = degree_inv.dot(attention_matrix_array)
                else:
                    attention_weights = attention_matrix
                    if remove_dia:
                        np.fill_diagonal(attention_matrix_array, 0)

                # for a in range(env.n_agents):
                # print (obses.shape)
                # all_obs = env.get_agent_obs()
                # all_obs = np.array(all_obs)
                # all_encoding = algo.policy.get_encoder_embedding(obses)
                if not de:
                    # all_encoding = algo.policy.get_embeddings(obses)
                    all_encoding = algo.policy.get_embeddings_nodes(obses)
                    # all_encoding = algo.policy.get_obses(obses)
                if de:
                    # all_encoding = algo.baseline.get_embeddings(obses)
                    all_encoding = algo.baseline.get_embeddings_nodes(obses)
                    # all_encoding = algo.baseline.get_obses(obses)

                # print(all_encoding.shape)

                # print(all_obs[0])
                # print("")
                # print (all_obs.shape)

                for a in range(n_agents):
                    if normalize:
                        attention_weights_0 = normalized_attention_weights[a]
                    else:
                        attention_weights_0 = attention_weights[a]

                    for i_agent in range(env.n_agents):
                        # remove the self-weights
                        if remove_dia:
                            if a == i_agent:
                                continue

                        # d = np.dot(all_obs[a], all_obs[i_agent])
                        # d = np.dot(all_obs[a], all_obs[i_agent])

                        # d = cosine_similarity(all_obs[a], all_obs[i_agent])
                        d = cosine_similarity(all_encoding[a], all_encoding[i_agent])
                        
                        d_1 = math.floor(d * 20) / 20 + 0.025
                        if d_1 > 1:
                            d_1 = 1
                        if d<-1:
                            d_1=-1

                        # print (all_obs[a], all_obs[i_agent])
                        # print (env.agent_pos[a], env.agent_pos[i_agent])
                        # print ("")

                        phy_d = np.sqrt(
                            (env.agent_pos[a][0] - env.agent_pos[i_agent][0]) ** 2
                            + (env.agent_pos[a][1] - env.agent_pos[i_agent][1]) ** 2)

                        if d_1 not in distance_vs_weight.keys():
                            distance_vs_weight[d_1] = [attention_weights_0[i_agent]]
                        else:
                            distance_vs_weight[d_1].append(attention_weights_0[i_agent])

                        if phy_d not in phy_d_vs_similarity.keys():
                            phy_d_vs_similarity[phy_d] = [d]
                        else:
                            phy_d_vs_similarity[phy_d].append(d)

                        if phy_d not in phy_d_vs_weight.keys():
                            phy_d_vs_weight[phy_d] = [attention_weights_0[i_agent]]
                        else:
                            phy_d_vs_weight[phy_d].append(attention_weights_0[i_agent])

                if plotMatrix:
                    plt.figure(figsize=(8, 6))
                    title = str(seed) + "-" + str(i_eps) + "-" + str(i_step)
                    if normalize:
                        sns.heatmap(normalized_attention_weights, annot=True, cmap="crest",
                                    fmt=".2f")  # annot显示数值，fmt设置数值格式
                    else:
                        sns.heatmap(attention_weights, annot=True, cmap="crest",
                                    fmt=".2f")  # annot显示数值，fmt设置数值格式
                    plt.title(title, fontsize=16)

                    case_study_dir = os.path.join(results_folder, "case_study/")
                    os.makedirs(case_study_dir, exist_ok=True)

                    plt.savefig(case_study_dir +"m={}_penalty={}_seedEpiStep={}.pdf".format(args.attention_type,
                                                                                    args.penalty, title),
                                bbox_inches="tight")  # dpi提高清晰度，bbox_inches防止截断
                    plt.close()

                obses, _, agent_dones, _ = env.step(actions)

                if agent_dones:
                    if i_step < args.max_env_steps - 1:
                        print('eps {} captured all preys in {} steps'.
                              format(i_eps + 1, i_step + 1))
                    break
            env.close()


    settings = "nor={}_nagents={}".format(normalize,n_agents)
    y_type = 'Avg. Attention Weight'
    x_type = 'Cosine Similarity'
    # print(distance_vs_weight)
    # distance_vs_weight = aggregate_dict_by_intervals(distance_vs_weight)
    # print(distance_vs_weight)
    plot_attn_stats_simple(distance_vs_weight, results_folder, args.penalty, len(matching_folders), de, args.attention_type, settings, x_type,y_type)

    y_type = 'Avg. Cosine Similarity'
    x_type = 'Position Distance'
    plot_attn_stats_simple(phy_d_vs_similarity, results_folder, args.penalty, len(matching_folders), de, args.attention_type, settings, x_type,y_type)

    y_type = 'Avg. Attention Weight'
    x_type = 'Position Distance'
    plot_attn_stats_simple(phy_d_vs_weight, results_folder, args.penalty, len(matching_folders), de, args.attention_type, settings, x_type, y_type)

    for i in matching_folders:
        print (i)