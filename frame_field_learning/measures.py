import torch


def iou(y_pred, y_true, threshold):
    assert len(y_pred.shape) == len(y_true.shape) == 2, "Input tensor shapes should be (N, .)"
    ignore_index = 255
    
    mask = y_true != 255
    mask_pred = (threshold < y_pred) & mask
    mask_true = threshold < y_true
    # mask = (y_true != 255).int()    # こっちだと、union=tensor([6, 3, 0])
    # mask_pred = (threshold < y_pred) * mask

    # print(f"{mask=}")
    # print(f"{mask_pred=}")
    # print(f"{mask_true=}")

    # count = torch.sum(y_true == 255).item()
    # print(f"{count=}")

    intersection = torch.sum(mask_pred * mask_true, dim=-1)
    # print(f"{intersection=}")  # tensor([3, 0, 0])
    union = torch.sum(mask_pred + mask_true, dim=-1)
    # print(f"{union=}")  # tensor([3, 3, 0])
    r = intersection.float() / union.float()
    r[union == 0] = 1
    return r


def dice_loss(y_pred, y_true, smooth=1, eps=1e-7):
    """

    @param y_pred: (N, C, H, W)
    @param y_true: (N, C, H, W)
    @param smooth:
    @param eps:
    @return: (N, C)
    """

    # ignore_index = 255
    # mask = (y_true != ignore_index).int()
    # masked_pred = y_pred * mask

    # print(f"{y_pred=}")
    # print(f"{masked_pred=}")

    numerator = 2 * torch.sum(y_true * y_pred, dim=(-1, -2))
    denominator = torch.sum(y_true, dim=(-1, -2)) + torch.sum(y_pred, dim=(-1, -2))
    return 1 - (numerator + smooth) / (denominator + smooth + eps)


def main():
    # import kornia
    # spatial_gradient_function = kornia.filters.SpatialGradient()
    #
    # image = torch.zeros((7, 7))
    # image[2:5, 2:5] = 1
    # print(image)
    #
    # grads = spatial_gradient_function(image[None, None, ...])[0, 0, ...] / 4
    # print(grads[0])
    # print(grads[1])

    # default array
    # y_true = torch.tensor([
    #     [0, 0, 0, 0, 1, 1, 1],
    #     [0, 0, 0, 0, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0]
    # ])
    # y_pred = torch.tensor([
    #     [0, 0, 0, 0, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    # ])
    # array for ignore
    y_true = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [255, 255, 255, 0, 0, 0, 0]
    ])
    y_pred = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
    ])
    print(y_true.shape)
    print(y_pred.shape)
    r = iou(y_pred, y_true, threshold=0.5)
    # default:          tensor([1., 0., 1.])
    # array for ignore: tensor([1., 0., 1.])
    print(r)
    print(torch.mean(r))

    dice_loss(y_pred, y_true)


if __name__ == "__main__":
    main()