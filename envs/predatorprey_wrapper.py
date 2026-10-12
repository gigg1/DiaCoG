# Using local gym
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

from PIL import Image, ImageDraw, ImageFont
import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from envs.ma_gym.envs.predator_prey import PredatorPrey
import gym
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageColor

from .utils import standard_eval

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}

class PredatorPreyWrapper(PredatorPrey):

    def __init__(self, centralized, other_agent_visible=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent_visible = other_agent_visible
        self.action_space = self.action_space[0]
        if self._agent_visible:
            mask_size = np.prod(self._agent_view_mask)
            self._obs_low = np.array([0., 0.] + [0.] * mask_size * 2 + [0.0])
            self._obs_high = np.array([1., 1.] + [1.] * mask_size * 2 + [1.0])
            self.observation_space = gym.spaces.Box(self._obs_low, self._obs_high)
        else:
            self.observation_space = self.observation_space[0]
        self.centralized = centralized
        if centralized:
            self.observation_space = gym.spaces.Box(
                low=np.array(list(self.observation_space.low) * self.n_agents),
                high=np.array(list(self.observation_space.high) * self.n_agents)
            )
        self.pickleable = True

    def get_avail_actions(self):
        avail_actions = [[1] * self.action_space.n for _ in range(self.n_agents)]
        if not self.centralized:
            return avail_actions
        else:
            return np.concatenate(avail_actions)

    def get_agent_obs(self):
        obs = super().get_agent_obs()
        if self._agent_visible:
            for i_agent in range(self.n_agents):
                pos = self.agent_pos[i_agent]
                # check if other agents are in the view area
                _agent_pos = np.zeros(self._agent_view_mask)
                for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                    for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                        if PRE_IDS['agent'] in self._full_obs[row][col]:
                            # get relative position for the prey loc:
                            _agent_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1

                obs[i_agent].extend(_agent_pos.flatten().tolist())
        return obs

    def step(self, actions):
        obses, rewards, dones, infos = super().step(actions)
        # print ("rewards:", rewards)
        # print(" np.mean(rewards):",  np.mean(rewards))

        if not self.centralized:
            return obses, rewards, dones, infos
        else:
            # print("---------------------")
            # print("np.concatenate(obses).shape:", np.concatenate(obses).shape)
            return np.concatenate(obses), np.mean(rewards), np.all(dones), infos

    def reset(self):
        obses = super().reset()
        if not self.centralized:
            return obses
        else:
            return np.concatenate(obses)

    # def eval(self, policy, n_episodes=20, greedy=True, load_from_file=False,
    #          render=False):
    #     standard_eval(self, policy, n_episodes=n_episodes, greedy=greedy,
    #         load_from_file=load_from_file, render=render)

    def eval(self, policy, n_episodes=20, greedy=True, load_from_file=False,
             render=False):
        avg_return = standard_eval(self, policy, n_episodes=n_episodes, greedy=greedy,
            load_from_file=load_from_file, render=render)
        return avg_return

    def my_render(self, attention_weights=None, cell_size=35):
        if attention_weights is not None:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = Image.fromarray(self.render(mode='rgb_array'))
            
            # Plot attention weights for agent 0

            start_col, start_row = self.agent_pos[0]
            start_x, start_y = (start_row + 0.5) * cell_size, (start_col + 0.5) * cell_size
            for i in range(self.n_agents):
                if i == 0:
                    draw_circle(
                        img, self.agent_pos[i], cell_size=cell_size, 
                        fill=None, outline='green', radius=0.1,
                        width=int(20 * attention_weights[i]))
                else:
                    if attention_weights[i] == 0:
                        fill = None
                    else:
                        fill = 'green'
                    end_col, end_row = self.agent_pos[i]
                    end_x, end_y = (end_row + 0.5) * cell_size, (end_col + 0.5) * cell_size
                    ImageDraw.Draw(img).line(((start_x, start_y), (end_x, end_y)), 
                        fill=fill, width=int(20 * attention_weights[i]))

            img = np.asarray(img)
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            self.render(mode='human')
    def my_render4(self, i_eps, step_count, attention_weights=None, cell_size=35, is_final=False,
                   save_images=False, output_base_dir="renders", cosine_similarity = None, start_agent = None):
        """
        增强的渲染函数，支持按episode保存图像

        参数:
            i_eps: episode编号
            step_count: 当前步数
            attention_weights: 注意力权重列表
            cell_size: 单元格大小
            is_final: 是否为最终渲染
            save_images: 是否保存图像
            output_base_dir: 保存图像的基目录
        """

        # 初始化图形系统
        # if not hasattr(self, '_render_fig'):
        #     self._init_render_system()

        # 创建episode特定目录
        if save_images:
            # 创建基目录
            os.makedirs(output_base_dir, exist_ok=True)
            # 创建episode特定目录
            episode_dir = os.path.join(output_base_dir, f"episode_{i_eps:03d}")
            os.makedirs(episode_dir, exist_ok=True)
        else:
            episode_dir = None
            return

        fig, self._render_ax = plt.subplots(figsize=(10, 10), dpi=100, facecolor='white')
        self._render_ax.set_axis_off()

        # 清除轴内容
        # self._render_ax.clear()
        # self._render_ax.set_axis_off()

        # 获取并显示基础渲染
        img = Image.fromarray(self.render(mode='rgb_array'))
        self._render_ax.imshow(np.asarray(img))
        # ax.imshow(np.asarray(img))

        # 绘制注意力权重
        if attention_weights is not None and start_agent is not None:
            start_col, start_row = self.agent_pos[start_agent]
            start_x, start_y = (start_row + 0.5) * cell_size, (start_col + 0.5) * cell_size

            # 绘制连接线
            for i in range(0, self.n_agents):
                if attention_weights[i] == 0 or start_agent == i:
                    continue

                end_col, end_row = self.agent_pos[i]
                end_x, end_y = (end_row + 0.5) * cell_size, (end_col + 0.5) * cell_size

                # line_width = max(1, 10 * attention_weights[i])
                line_width = max(1, 20 * attention_weights[i])
                alpha = min(0.9, 0.3 + 0.7 * attention_weights[i])
                color = "teal"

                # 绘制线条
                self._render_ax.plot(
                # ax.plot(
                    [start_x, end_x], [start_y, end_y],
                    color=color,
                    linewidth=line_width,
                    alpha=alpha,
                    solid_capstyle='round',
                    solid_joinstyle='round',
                    path_effects=[
                        patheffects.Stroke(linewidth=line_width + 1,
                                           foreground='black', alpha=0.3 * alpha),
                        patheffects.Normal()
                    ]
                )

        # 绘制智能体标记
        for i in range(self.n_agents):
            col, row = self.agent_pos[i]
            x, y = (row + 0.5) * cell_size, (col + 0.5) * cell_size

            if i == start_agent:
                # 主智能体
                circle = plt.Circle(
                    (x, y), cell_size * 0.35,
                    edgecolor='navy',
                    facecolor='blue',
                    linewidth=4,
                    alpha=1
                )
                self._render_ax.add_patch(circle)

                # 光晕效果
                glow = plt.Circle(
                    (x, y), cell_size * 0.45,
                    edgecolor='none',
                    facecolor='navy',
                    alpha=0.35
                )
                self._render_ax.add_patch(glow)
            else:
                # 其他智能体
                circle = plt.Circle(
                    (x, y), cell_size * 0.35,
                    edgecolor='white',
                    facecolor='blue',
                    linewidth=4,
                    alpha=1
                )
                self._render_ax.add_patch(circle)

            if attention_weights is not None:
                weight = attention_weights[i]
                # if weight > 0.1:
                self._render_ax.text(
                    x, y - cell_size * 0.3, f"w:{weight:.2f}",
                    color='white',
                    fontsize=12,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                )

            if cosine_similarity is not None:
                cs = cosine_similarity[i]
                self._render_ax.text(
                    x, y + cell_size * 0.3, f"cs:{cs:.2f}",
                    color='white',
                    fontsize=12,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                )

            # 添加智能体编号
            self._render_ax.text(
                x, y + cell_size * 0.035, str(i + 1),
                color='white',
                fontsize=20,
                ha='center',
                va='center',
                fontweight='bold',
            )

        # 更新标题
        self._render_ax.set_title(
            f"Episode: {i_eps} - Step: {step_count}",
            fontsize=18,
            color='black'
        )

        # # 动态更新图形
        # self._render_fig.canvas.draw_idle()
        # self._render_fig.canvas.flush_events()
        #
        # # 保存图像
        # if save_images and episode_dir:
        #     self._save_render_frame(i_eps, step_count, episode_dir, start_agent)
        #
        # # 固定帧率控制
        # # plt.pause(0.08)
        # plt.pause(0.01)
        #
        # return plt.fignum_exists(self._render_fig.number)

        # 保存图像
        if save_images and episode_dir:
            if start_agent is not None:
                filename = f"ep_{i_eps:03d}_step_{step_count:03d}_starta_{start_agent + 1:01d}.pdf"
            else:
                filename = f"ep_{i_eps:03d}_step_{step_count:03d}_final.pdf"
            filepath = os.path.join(episode_dir, filename)
            fig.savefig(
                filepath,
                dpi=200,
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor=fig.get_facecolor()
            )
            print(f"已保存: {filepath}")

        # 关闭图形释放内存
        plt.close(fig)

    def my_render_continueGame(self, i_eps, step_count, cell_size=35):
        # 获取基础渲染
        img = Image.fromarray(self.render(mode='rgb_array'))
        img = np.asarray(img)

        import matplotlib.pyplot as plt

        # 检查是否已经创建了图形，如果没有则创建
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(8, 8), dpi=100)
            self.ax = self.fig.add_subplot(111)
            plt.ion()  # 开启交互模式
        else:
            # 清除之前的内容
            self.ax.clear()

        self.ax.set_axis_off()
        # 显示基础环境
        self.ax.imshow(img)

        # 绘制智能体
        for i in range(self.n_agents):
            col, row = self.agent_pos[i]
            x, y = (row + 0.5) * cell_size, (col + 0.5) * cell_size
            # 其他智能体
            circle = plt.Circle((x, y), cell_size * 0.25,
                                    edgecolor='blue',
                                    facecolor='blue',
                                    linewidth=2,
                                    alpha=0.8)
            self.ax.add_patch(circle)
            # 添加标签
            self.ax.text(x, y, f'A{i}',
                             color='white',
                             fontsize=8,
                             ha='center',
                             va='center')

        # 添加标题显示当前step
        step_num = step_count
        episode_num = i_eps
        self.ax.set_title(f"Episode {episode_num}, Step {step_num}", fontsize=16)

        # 更新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 短暂暂停，让画面更新
        plt.pause(0.001)

        # 返回viewer状态
        if hasattr(self, 'viewer') and self.viewer is not None:
            return self.viewer.isopen
        else:
            return True

    def is_neighbor(self, i, j):
        pos_i = self.agent_pos[i]
        pos_j = self.agent_pos[j]
        if np.abs(pos_i[0] - pos_j[0]) <= 1 and np.abs(pos_i[1] - pos_j[1]) <= 1:
            return True
        else:
            return False

    def get_proximal_adjacency_matrix(self, obs_n, dec_obs_dim):
        obs_n_original_shape = obs_n.shape
        obs_n = obs_n.reshape(-1, self.n_agents, dec_obs_dim)
        self_connected_adj = \
            torch.zeros(obs_n.shape[:-2] + (self.n_agents, self.n_agents))
        D = torch.zeros_like(self_connected_adj)
        
        if len(obs_n.shape) == 2:
            for i_agent in range(self.n_agents):
                # Self connection
                self_connected_adj[i_agent, i_agent] = 1
                D[i_agent, i_agent] += 1
                for j_agent in range(i_agent + 1, self.n_agents):
                    if self.is_neighbor(i_agent, j_agent):
                        self_connected_adj[i_agent, j_agent] = 1
                        self_connected_adj[j_agent, i_agent] = 1
                        D[i_agent, i_agent] += 1
                        D[j_agent, j_agent] += 1

            D = D.diagonal().pow(-0.5).diag()

        else:
            for i_step in range(int(obs_n.shape[0])):
                for i_agent in range(self.n_agents):
                    # Self connection
                    self_connected_adj[i_step, i_agent, i_agent] = 1
                    D[i_step, i_agent, i_agent] += 1
                    for j_agent in range(i_agent + 1, self.n_agents):
                        if self.is_neighbor(i_agent, j_agent):
                            self_connected_adj[i_step, i_agent, j_agent] = 1
                            self_connected_adj[i_step, j_agent, i_agent] = 1
                            D[i_step, i_agent, i_agent] += 1
                            D[i_step, j_agent, j_agent] += 1
                
                D[i_step, :, :] = D[i_step, :, :].diagonal().pow(-0.5).diag()

            self_connected_adj = self_connected_adj.reshape(
                obs_n_original_shape[:-2] + (self.n_agents, self.n_agents))
            D = D.reshape(
                obs_n_original_shape[:-2] + (self.n_agents, self.n_agents))

            processed_self_connected_adj = torch.matmul(D, self_connected_adj)
            processed_self_connected_adj = torch.matmul(processed_self_connected_adj, D)

        return self_connected_adj, processed_self_connected_adj



def draw_circle(image, pos, cell_size=50, fill='white', outline='black', 
                radius=0.3, width=1):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * radius
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], fill=fill, 
        outline=outline, width=width)


if __name__ == '__main__':
    env = PredatorPreyWrapper(centralized=True)
    print(env.action_space)
    print(env.observation_space)