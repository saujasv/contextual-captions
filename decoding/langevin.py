import torch
from utils import get_input_ids


def pNCG(
    context,
    target,
    seq_length,
    energy_function,
    embedding_layer,
    n_iterations,
    alpha,
    P,
    tokenizer,
    device,
    init_state=None,
):
    batch_size=1
    grad_energy = torch.func.grad(energy_function)
    embedding_matrix = embedding_layer.weight.unsqueeze(0).unsqueeze(0)
    x = (
        embedding_layer(
            torch.randint(0, embedding_layer.num_embeddings, (batch_size, seq_length)).to(device)
        )
        if init_state is None
        else init_state
    )


    for n in range(n_iterations):
        tokens = get_input_ids(embedding_layer.weight, x)
        log_p_x = energy_function(x, context, target)
        print(log_p_x.item())
        grad_x = grad_energy(x, context, target)

        D = embedding_matrix.expand((batch_size, seq_length, -1, -1)) - x.unsqueeze(2)
        proposal_forward_unnorm = torch.exp(
            -(1 / 2) * torch.einsum("bsd,bsvd->bsv", grad_x, D)
            - (1 / (2 * alpha)) * (torch.pow(D.norm(dim=-1, p=P), P))
        )
        proposal_forward = proposal_forward_unnorm / proposal_forward_unnorm.sum(
            dim=-1, keepdim=True
        )

        tokens_next = (
            torch.multinomial(proposal_forward_unnorm.squeeze(), 1)
            .squeeze()
            .unsqueeze(0)
        )

        #tokens_next = (
        #    torch.argmax(proposal_forward_unnorm, dim=2)
        #)
        prob_forward = torch.gather(proposal_forward_unnorm, 2, tokens_next.unsqueeze(2))
        x_next = embedding_layer(tokens_next)

        #print(tokens,tokens_next)
        print(tokenizer.batch_decode(tokens), tokenizer.batch_decode(tokens_next))

        x_next_mult = x.clone().unsqueeze(1).expand(-1, seq_length, -1, -1)
        x_next_mult[:, range(seq_length), range(seq_length), :] = x_next
        x_next_mult = x_next_mult.reshape(-1, x_next_mult.shape[2], x_next_mult.shape[3])

        log_p_x_next = energy_function(x_next_mult, context, target).reshape(batch_size, seq_length, -1)
        #print(log_p_x_next)
        grad_x_next = grad_energy(x_next, context, target)
        D_next = embedding_matrix.expand((batch_size*seq_length, seq_length, -1, -1)) - x_next_mult.unsqueeze(2)

        proposal_backward_unnorm = torch.exp(
            -(1 / 2) * torch.einsum("bsd,bsvd->bsv", grad_x_next, D_next)
            - (1 / (2 * alpha)) * (torch.pow(D_next.norm(dim=-1, p=P), P))
        )

        proposal_backward = proposal_backward_unnorm / proposal_backward_unnorm.sum(
            dim=-1, keepdim=True
        )
        prob_backward = torch.gather(proposal_backward_unnorm, 2, tokens.unsqueeze(2))

        acceptance_probs = (
            (prob_backward.log() - prob_forward.log()) + (log_p_x - log_p_x_next)
        ).exp().squeeze(-1)
        #print(acceptance_probs)
        p = torch.rand(acceptance_probs.shape, device=acceptance_probs.device)
        accepted_transitions = torch.where(
            p < torch.min(torch.ones_like(acceptance_probs), acceptance_probs)
        )[1]

        x[:, accepted_transitions, :] = x_next[:, accepted_transitions, :]
        print("#accepted =", len(accepted_transitions))

        # accept_prob = torch.min(
        #     (
        #         (prob_backward.log().sum() - prob_forward.log().sum())
        #         + (log_p_x_next - log_p_x)
        #     ).exp(),
        #     torch.tensor(1.0, device=device),
        # )

        # if torch.rand(1, device=device) < accept_prob:
        #     x = torch.Tensor(x_next)
        #     print("Accepted, prob =", accept_prob.item())
        # else:
        #     print("Rejected, prob =", accept_prob.item())

        del (
            x_next,
            tokens_next,
            log_p_x,
            log_p_x_next,
            grad_x,
            grad_x_next,
            D,
            D_next,
            proposal_backward_unnorm,
            proposal_backward,
            prob_backward,
            proposal_forward_unnorm,
            proposal_forward,
            prob_forward,
            # acceptance_probs,
            # p,
        )

    return get_input_ids(embedding_layer.weight, x)


if __name__ == "__main__":
    from transformers import (
        Blip2Processor,
        Blip2ForConditionalGeneration,
        GPT2LMHeadModel,
        GPT2Tokenizer,
    )
    from speaker import Blip2Speaker, GPT2Speaker
    import torch
    from PIL import Image
    from pathlib import Path

    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

    # model = Blip2ForConditionalGeneration.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b-coco",
    #     device_map="cuda:0",
    #     torch_dtype=torch.bfloat16,
    # )

    # speaker = Blip2Speaker(model, processor)

    img_set = "open-images-2057_2fc6afbbb663b164"
    image_path = Path(
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets"
    )
    images = [Image.open(image_path / img_set / f"img{i}.jpg") for i in range(10)]

    # inputs = processor(images=images[5], return_tensors="pt").to(model.device)
    # outputs = model.generate(
    #     **inputs, do_sample=True, output_scores=True, return_dict_in_generate=True
    # )

    # init_seq = torch.randint(
    #     0, model.language_model.get_input_embeddings().num_embeddings, (1, 32)
    # ).to(model.device)
    # init_state = model.language_model.get_input_embeddings()(init_seq)
    # init_state[0, : outputs.sequences.shape[1], :] = (
    #     model.language_model.get_input_embeddings()(outputs.sequences[0])
    # )
    # print(init_state.shape)

    # print(
    #     torch.gather(
    #         torch.stack(outputs.scores, dim=1), 2, outputs.sequences.unsqueeze(2)
    #     )
    #     .log_softmax(dim=-1)
    #     .sum()
    # )

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", device_map="cuda:0", torch_dtype=torch.float32
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    speaker = GPT2Speaker(model, tokenizer)

    init_state = torch.tensor(tokenizer('He was waiting at the gym')['input_ids'], dtype=torch.int).unsqueeze(0).to(model.device)

    print(init_state)
    sample = pNCG(
        images,
        10,
        6,
        speaker.energy,
        model.get_input_embeddings(),
        100,
        1,
        10,
        tokenizer,
        model.device,
        init_state=model.get_input_embeddings()(init_state),
    )
    print(tokenizer.batch_decode(sample, skip_special_tokens=True))