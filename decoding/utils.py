import torch


def get_input_ids(embeddings: torch.Tensor, input_embeds: torch.Tensor):
    # Solution due to https://discuss.pytorch.org/t/reverse-nn-embedding/142623/8
    embedding_matrix_size = input_embeds.size(0), input_embeds.size(1), -1, -1
    input_embeds_size = -1, -1, embeddings.size(0), -1
    input_ids = torch.argmin(
        torch.abs(
            input_embeds.unsqueeze(2).expand(input_embeds_size)
            - embeddings.unsqueeze(0).unsqueeze(0).expand(embedding_matrix_size)
        ).sum(dim=3),
        dim=2,
    )

    return input_ids
