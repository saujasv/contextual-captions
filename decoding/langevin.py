import torch
from .utils import get_input_ids
from tqdm import tqdm


def pNCG(
    context,
    target,
    seq_length,
    energy_function,
    listener_energy_function,
    embedding_layer,
    n_iterations,
    alpha,
    P,
    k,
    mix_lambda,
    tokenizer,
    device,
    init_state=None,
):
    batch_size = 1
    grad_energy = torch.func.grad(energy_function)
    listener_grad_energy = torch.func.grad(listener_energy_function)
    embedding_matrix = embedding_layer.weight.unsqueeze(0).unsqueeze(0)

    x = (
        embedding_layer(
            torch.randint(
                0, embedding_layer.num_embeddings, (batch_size, seq_length)
            ).to(device)
        )
        if init_state is None
        else init_state
    )

    # x[:, :init_state.shape[1]] = init_state

    for n in tqdm(range(n_iterations)):
        tokens = get_input_ids(embedding_layer.weight, x)
        log_p_x = energy_function(x, context, target)
        print(log_p_x.item())
        grad_x_speaker = grad_energy(x, context, target).detach()
        grad_x_listener = listener_grad_energy(x, context, target).detach()
        grad_x = mix_lambda * grad_x_speaker + (1 - mix_lambda) * grad_x_listener

        D = embedding_matrix.expand((batch_size, seq_length, -1, -1)) - x.unsqueeze(2)

        proposal_forward_unnorm = torch.exp(
            -(1 / 2) * torch.einsum("bsd,bsvd->bsv", grad_x, D)
            - (1 / (2 * alpha)) * (torch.pow(D.norm(dim=-1, p=P), P))
        )

        proposal_forward_thres = torch.zeros_like(proposal_forward_unnorm)

        topkf = torch.topk(proposal_forward_unnorm, k, dim=2)
        proposal_forward_thres[
            torch.arange(proposal_forward_thres.shape[0]).unsqueeze(-1).unsqueeze(-1),
            torch.arange(proposal_forward_thres.shape[1]).unsqueeze(0).unsqueeze(-1),
            topkf.indices,
        ] = topkf.values

        proposal_forward = proposal_forward_thres

        # TOP-K
        tokens_next = (
            torch.multinomial(proposal_forward_thres.squeeze(), 1)
            .squeeze()
            .unsqueeze(0)
        )

        # GREEDY
        # tokens_next = (
        #    torch.argmax(proposal_forward_unnorm, dim=2)
        # )

        prob_forward = torch.gather(proposal_forward, 2, tokens_next.unsqueeze(2))
        x_next = embedding_layer(tokens_next)

        # print(tokens,tokens_next)
        print(tokenizer.batch_decode(tokens), tokenizer.batch_decode(tokens_next))

        # x_next_mult = x.clone().unsqueeze(1).expand(-1, seq_length, -1, -1).clone()
        # x_next_mult[range(batch_size), range(seq_length), range(seq_length)] = x_next
        # x_next_mult = x_next_mult.reshape(-1, x_next_mult.shape[2], x_next_mult.shape[3])

        # log_p_x_next = energy_function(x_next_mult, context, target)
        # log_p_x_next = log_p_x_next.reshape(batch_size, seq_length, -1)
        # grad_x_next = grad_energy(x_next_mult, context, target)

        prob_backward = []
        log_p_x_next = []

        # D_next = embedding_matrix.expand((batch_size*seq_length, seq_length, -1, -1)) - x_next_mult.unsqueeze(2)
        # print(D_next.shape)

        for i in range(seq_length):

            x_next_mult = x.clone()
            x_next_mult[:, i, :] = x_next[:, i, :]

            log_p_x_mult_next = energy_function(x_next_mult, context, target).detach()
            log_p_x_next.append(log_p_x_mult_next)

            grad_x_next_mult_speaker = grad_energy(
                x_next_mult, context, target
            ).detach()
            grad_x_next_mult_listener = listener_grad_energy(
                x_next_mult, context, target
            ).detach()
            grad_x_next_mult = (
                mix_lambda * grad_x_next_mult_speaker
                + (1 - mix_lambda) * grad_x_next_mult_listener
            )

            # D_next = embedding_matrix.expand((batch_size*seq_length, seq_length, -1, -1)) - x_next_mult.unsqueeze(2)

            D_next = (
                embedding_matrix.expand((batch_size, seq_length, -1, -1))
                - x_next_mult.unsqueeze(2)
            ).detach()

            # proposal_backward_unnorm = torch.exp(
            #    - (1/ 2) * torch.einsum("bsd,bsvd->bsv", grad_x_next, D_next)
            #    - (1 / (2 * alpha)) * (torch.pow(D_next.norm(dim=-1, p=P), P))
            # )

            # proposal_backward_unnorm = torch.stack([torch.exp(
            #    - (1/ 2) * torch.einsum("sd,svd->sv", grad_x_next[i], D_next[i])
            #    - (1 / (2 * alpha)) * (torch.pow(D_next.norm(dim=-1, p=P), P))
            # ) for i in range(grad_x_next.shape[0])]).squeeze()

            proposal_backward_unnorm = torch.exp(
                -(1 / 2) * torch.einsum("bsd, bsvd -> bsv", grad_x_next_mult, D_next)
                - (1 / (2 * alpha)) * (torch.pow(D_next.norm(dim=-1, p=P), P))
            )

            proposal_backward = proposal_backward_unnorm
            proposal_backward = torch.gather(proposal_backward, 2, tokens.unsqueeze(2))
            # proposal_backward = proposal_backward[tokens[:, i], :]

            prob_backward.append(proposal_backward[:, i, :].detach())
            # prob_backward = torch.gather(proposal_backward, 2, tokens.unsqueeze(2))

            del D_next
            del x_next_mult
            del grad_x_next_mult

        prob_backward = torch.stack(prob_backward, dim=1)
        log_p_x_next = torch.stack(log_p_x_next)

        acceptance_probs = (
            (
                (prob_backward.squeeze().log() - prob_forward.squeeze().log())
                + (log_p_x - log_p_x_next)
            )
            .exp()
            .squeeze(-1)
        )

        # print(prob_backward.log(), prob_forward.log(), log_p_x, log_p_x_next)

        # print(acceptance_probs)

        p = torch.rand(acceptance_probs.shape, device=acceptance_probs.device)
        accepted_transitions = torch.where(
            p < torch.min(torch.ones_like(acceptance_probs), acceptance_probs)
        )[0]

        x[:, accepted_transitions, :] = x_next[:, accepted_transitions, :]
        # x = x_next
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
            topkf,
            x_next,
            tokens_next,
            log_p_x,
            log_p_x_next,
            # log_p_x_next_mult,
            grad_x,
            # grad_x_next,
            # grad_x_mult_next,
            # x_next_mult,
            D,
            # D_next,
            proposal_backward_unnorm,
            proposal_backward,
            prob_backward,
            proposal_forward_unnorm,
            proposal_forward_thres,
            proposal_forward,
            prob_forward,
            acceptance_probs,
            p,
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
    from listener import Blip2Listener
    import torch
    from PIL import Image
    from pathlib import Path

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    speaker = Blip2Speaker(model, processor)
    listener = Blip2Listener(model, processor)

    img_set = "open-images-2057_2fc6afbbb663b164"
    image_path = Path(
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets"
    )
    images = [Image.open(image_path / img_set / f"img{i}.jpg") for i in range(10)]

    inputs = processor(images=images[5], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=32,
    )

    print(processor.batch_decode(outputs.sequences[0]))
    init_state = outputs.sequences[0].unsqueeze(0)
    print(outputs.sequences[0])
    cutoff = (
        (outputs.sequences[0] == 50140).nonzero()[0]
        if torch.any(outputs.sequences[0] == 50140)
        else outputs.sequences[0].shape[0]
    )
    print(cutoff)
    init_state = init_state[:, :cutoff]

    # init_seq = torch.randint(
    #    0, model.language_model.get_input_embeddings().num_embeddings, (1, 16)
    # ).to(model.device)
    # print(init_seq.shape)
    # print(outputs.sequences[0].shape)
    # init_state = model.language_model.get_input_embeddings()(init_seq)
    # init_state[0, : outputs.sequences.shape[1], :] = (
    #    model.language_model.get_input_embeddings()(outputs.sequences[0])
    # )
    # print(init_state.shape)

    # print(
    #     torch.gather(
    #         torch.stack(outputs.scores, dim=1), 2, outputs.sequences.unsqueeze(2)
    #     )
    #     .log_softmax(dim=-1)
    #     .sum()
    # )

    # model = GPT2LMHeadModel.from_pretrained(
    #    "gpt2", device_map="cuda:0", torch_dtype=torch.bfloat16
    # )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # speaker = GPT2Speaker(model, tokenizer)

    init_state_2 = (
        torch.tensor(
            tokenizer(
                "A man in a closet eating a chicken and a doctor away from the evil children"
            )["input_ids"],
            dtype=torch.int,
        )
        .unsqueeze(0)
        .to(model.device)
    )

    print(init_state.shape)
    print(init_state_2.shape)

    # print(init_state)
    sample = pNCG(
        images,
        3,
        init_state.shape[1],
        speaker.energy,
        listener.energy,
        model.language_model.get_input_embeddings(),
        100,
        5.0,
        10,
        10,
        0.5,
        processor.tokenizer,
        model.device,
        init_state=model.language_model.get_input_embeddings()(init_state),
    )
    print(tokenizer.batch_decode(sample, skip_special_tokens=True))
