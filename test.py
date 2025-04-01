import torch

def main0():
    images = torch.randint(low = 0, high = 2, size = (1, 3, 2, 2))  # torch.Size([10, 3, 300, 300])

    # 黒(0, 0, 0)かどうかを判定
    binary_mask = (images.sum(dim=1) > 0).int()  # torch.Size([10, 300, 300])

    print(images)
    print(binary_mask.shape)  # torch.Size([10, 300, 300])
    print(binary_mask)



def main1():
    mask = torch.randn(10, 300, 300)  # 実際の mask を使用

    # 次元を追加して (10, 1, 300, 300) にし、その後 expand で (10, 3, 300, 300) にする
    mask_expanded = mask.unsqueeze(1).expand(-1, 3, -1, -1)

    print(mask_expanded.size())  # torch.Size([10, 3, 300, 300])


def main2():
    # 元のテンソルを作成
    tensorA = torch.tensor([10, 300, 300])

    # テンソルをコピー
    tensorB = tensorA.clone()

    # コピーしたテンソルの先頭要素を変更
    tensorB[0] = 20

    print("Original tensorA:", tensorA)  # tensor([ 10, 300, 300])
    print("Modified tensorB:", tensorB)  # tensor([ 20, 300, 300])

def main3():
    b = torch.randint(low = 0, high = 2, size = (2, 3, 3))
    print(b.shape)  # torch.Size([2, 3, 3])
    print(b)

    # b_repeated = b.repeat(2, 1, 1)  # 2倍にリピート(a, b, a, b)
    # print(b_repeated.shape)  # torch.Size([4, 3, 3])
    # print(b_repeated)

    transformed_b = torch.cat([b[0].repeat(2, 1, 1),
                                b[1].repeat(2, 1, 1)], dim=0)  # (a, a, b, b)
    # mask_expanded = torch.cat([mask[i].repeat(2, 1, 1) for i in range(10)], dim=0)

    print(transformed_b.shape)  # torch.Size([4, 3, 3])
    print(transformed_b)

def main4():
    gt_sum  = torch.tensor([[[1, 1, 1],
                            [0, 0, 0],
                            [0, 0, 0]]])
    mask  = torch.tensor([[[1, 1, 0],
                            [1, 1, 0],
                            [1, 1, 0]]])
    count = ((gt_sum == 1) & (mask == 0)).sum().item()

    print(gt_sum)
    print(mask)
    print(count)


if __name__ == "__main__":
    main4()