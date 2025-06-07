from visual import sat
import gymnasium as gym
import multiprocessing as mp
import numpy as np
from gymnasium.wrappers import ResizeObservation
from wrappers import FireEpisodicLifeEnv, ScaledFloatFrame


@sat("Initializing LimitedParallelEnv")
class LimitedParallelEnv:
    def __init__(self, env_name: str, n_envs: int, max_workers: int,
                 is_pixel_env: bool = False, is_fire: bool = False):
        self.env_name = env_name
        self.is_pixel_env = is_pixel_env
        self.is_fire = is_fire
        self.global_indices = list(range(n_envs))
        self.n_envs = n_envs
        self.max_workers = min(max_workers, n_envs)

        self.worker_groups = self._split_indices()
        self.pipes, self.procs = self._start_workers()

    def _split_indices(self):
        """Разбивает global_indices на примерно равные группы по числу воркеров."""
        group_size = int(np.ceil(self.n_envs / self.max_workers))
        return [self.global_indices[i:i+group_size]
                for i in range(0, self.n_envs, group_size)]

    def _start_workers(self):
        """Запускает процессы-воркеры и возвращает списки пайпов и процессов."""
        pipes, procs = [], []
        for group in self.worker_groups:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_worker,
                           args=(child_conn, self.env_name, group,
                                 self.is_pixel_env, self.is_fire),
                           daemon=True)
            p.start()
            child_conn.close()
            pipes.append(parent_conn)
            procs.append(p)
        return pipes, procs

    @sat("Resetting all environments")
    def reset(self):
        """Возвращает np.ndarray наблюдений в порядке global_indices."""
        obs_dict = self._gather_initial_observations()
        return self._dict_to_array(obs_dict)

    def _gather_initial_observations(self):
        """Получает начальные наблюдения от всех воркеров."""
        obs_dict = {}
        for pipe in self.pipes:
            for idx, o in pipe.recv():
                obs_dict[idx] = o
        return obs_dict

    def _dict_to_array(self, data_dict):
        """Конвертирует словарь {idx: value} в np.ndarray, упорядоченный по global_indices."""
        return np.array([data_dict[idx] for idx in self.global_indices])

    # @sat("Stepping through environments")
    def step(self, actions: np.ndarray):
        """Выполняет шаг для всех env, возвращает (obs, rewards, dones) как np.ndarray."""
        assert len(actions) == self.n_envs, "Неверная длина массива действий"
        action_map = {idx: actions[pos] for pos, idx in enumerate(self.global_indices)}

        self._send_steps(action_map)
        obs_dict, rew_dict, done_dict = self._collect_step_results()
        return (self._dict_to_array(obs_dict),
                self._dict_to_array(rew_dict),
                self._dict_to_array(done_dict))

    def _send_steps(self, action_map):
        """Отправляет батчи действий каждому воркеру."""
        for pipe, group in zip(self.pipes, self.worker_groups):
            batch = [(idx, action_map[idx]) for idx in group]
            pipe.send(("step", batch))

    def _collect_step_results(self):
        """Принимает результаты шага от воркеров и возвращает словари obs, rew, done."""
        obs_dict, rew_dict, done_dict = {}, {}, {}
        for pipe in self.pipes:
            for idx, (o, r, d) in pipe.recv():
                obs_dict[idx] = o
                rew_dict[idx] = r
                done_dict[idx] = d
        return obs_dict, rew_dict, done_dict

    def remove_envs(self, flags: np.ndarray):
        """Удаляет env, где flags == 0, обновляет группы и процессы."""
        flags = np.asarray(flags).flatten()
        assert len(flags) == self.n_envs, "flags должен совпадать по длине с числом сред"

        remaining = [idx for pos, idx in enumerate(self.global_indices) if flags[pos]]
        to_remove = set(self.global_indices) - set(remaining)

        self._send_remove_commands(to_remove)
        self._prune_workers(to_remove)
        self._update_indices(remaining)

    def _send_remove_commands(self, to_remove):
        """Посылает воркерам команды 'remove' для удаления конкретных idx."""
        for pipe, group in zip(self.pipes, self.worker_groups):
            local_remove = [idx for idx in group if idx in to_remove]
            if local_remove:
                pipe.send(("remove", local_remove))

    def _prune_workers(self, to_remove):
        """Закрывает воркерские процессы, если их группы пусты после удаления."""
        new_groups, new_pipes, new_procs = [], [], []
        for pipe, proc, group in zip(self.pipes, self.procs, self.worker_groups):
            updated = [idx for idx in group if idx not in to_remove]
            if updated:
                new_groups.append(updated)
                new_pipes.append(pipe)
                new_procs.append(proc)
            else:
                pipe.send(("close", None))
                pipe.close()
                proc.join()
        self.worker_groups, self.pipes, self.procs = new_groups, new_pipes, new_procs

    def _update_indices(self, remaining):
        """Обновляет global_indices и n_envs после удаления."""
        self.global_indices = remaining
        self.n_envs = len(self.global_indices)

    def close(self):
        self._send_close_commands()
        self._join_all_workers()

    def _send_close_commands(self):
        """Посылает комманду 'close' всем воркерам."""
        for pipe in self.pipes:
            pipe.send(("close", None))
            pipe.close()

    def _join_all_workers(self):
        """Дожидается завершения всех воркерских процессов."""
        for proc in self.procs:
            proc.join()


def _wrap_env(env_name, is_pixel, is_fire):
    """Создаёт базовую среду и применяет обёртки по флагам."""
    env = gym.make(env_name, render_mode='rgb_array')
    if is_fire:
        env = FireEpisodicLifeEnv(env)
    if is_pixel:
        env = ResizeObservation(env, (84, 84))
        env = ScaledFloatFrame(env)
    return env


def _worker(pipe, env_name, indices, is_pixel, is_fire):
    """Воркер хранит env для каждого своего global_idx и обрабатывает команды."""
    envs = {idx: _wrap_env(env_name, is_pixel, is_fire) for idx in indices}

    init = [(idx, env.reset()[0]) for idx, env in envs.items()]
    pipe.send(init)

    is_ant = 'AntBulletEnv' in env_name
    while True:
        cmd, data = pipe.recv()
        if cmd == "step":
            results = _process_step_batch(envs, data, is_ant)
            pipe.send(results)
        elif cmd == "remove":
            _process_remove(envs, data)
        elif cmd == "close":
            _process_close(envs)
            pipe.close()
            break
        else:
            raise ValueError(f"Unknown command {cmd}")


def _process_step_batch(envs, batch, is_ant):
    """Обрабатывает список (idx, action) и возвращает список (idx, (o, r, done))."""
    results = []
    for idx, action in batch:
        env = envs[idx]
        o, r, term, trunc, _ = env.step(action)
        if is_ant:
            r = env.unwrapped.rewards[1]
        done = term or trunc
        results.append((idx, (o, r, done)))
    return results


def _process_remove(envs, remove_list):
    """Закрывает и удаляет envs по списку global_idx."""
    for idx in remove_list:
        envs[idx].close()
        del envs[idx]


def _process_close(envs):
    """Закрывает все env, оставшиеся в словаре."""
    for env in envs.values():
        env.close()
