from atari_wrappers import WarpFrame



W = 64
H = 64


def wrap_env(env, width=84, height=84, gray_scale=False):
    env = WarpFrame(env, width, height, gray_scale)
    return env