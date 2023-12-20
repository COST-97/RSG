ENV_DESC = {
    "Indoor Floor": ((0.6, 0.9), 0.0, 0.0),
    "Ice Surface": ((0.01, 0.1), 0.0, 0.0),
    "UpStairs": ((1.2, 1.5), (0.0, 13.125), (0.0, 0.4)),
    "DownStairs": ((1.2, 1.5), (0.0, 14.375), (-0.26, 0.0)),
    "Marble Slope Uphill": ((0.7, 1.1), (2.25, 2.625), (0.15, 0.25)),
    "Marble Slope Downhill": ((0.7, 1.1), (3.0, 3.375), (-0.3, -0.18)),
    "Grassland": ((0.5, 0.7), (0.25, 9.0), 0.0),
    "Grassland Slope Uphill": ((0.5, 0.7), (0.25, 6.125), (0.06, 0.1)),
    "Grassland Slope Downhill": ((0.5, 0.7), (0.375 , 7.75), (-0.25, -0.15)),
    "Grass and Pebble": ((0.05, 0.1), (0.0 , 25.375), 0.0),
    "Steps": ((0.6, 1.2), (0.0 , 12.75), 0.0),
    "UpStairs": ((1.2, 1.5), (0.0, 13.125), (0.0, 0.4)),
    "Grass and Sand": ((0.3, 0.4), (0.25 , 5.625), 0.0),
    # "Hills": ((0.2, 0.3), (0.0 , 4.5), (-5,5)),
}

CN_DESC = {
    "Indoor Floor": "室内",
    "Ice Surface": "冰面",
    "UpStairs": "上行楼梯",
    "DownStairs": "下行楼梯",
    "Marble Slope Uphill": "上坡",
    "Marble Slope Downhill": "下坡",
    "Grassland": "草地",
    "Grassland Slope Uphill": "草地上坡",
    "Grassland Slope Downhill": "草地下坡",
    "Grass and Pebble": "石子草地",
    "Steps": "阶梯",
    "Grass and Sand": "沙子草地",

}

env_mapper = {k: i for i, k in enumerate(ENV_DESC.keys())}

ENV_NUM = len(ENV_DESC)
assert ENV_NUM == 12
ENV_DIM = 3
ENV_SAMPLE_NUM = 100