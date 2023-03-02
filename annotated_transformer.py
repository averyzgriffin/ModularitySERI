"""

1. model(x)
    > x: (b,3) e.g. 
    tensor([[ 34,  92, 113],
            [ 96,  94, 113],
            [ 92, 102, 113],
            [ 38,   5, 113],
            [ 59, 106, 113],
            [ 86,  87, 113]])

2. Embed(x) -> result = torch.einsum('dbp -> bpd', self.W_E[:, x])
    > W_E = (128,114)   each number gets a unique, learned embedding
    >this first grabs the columns of W_E corresponding to the numbers of x
    >e.g. columns 34, 92, 113 are grabbed for the first sample
    >einsum then rearranged the dimensions to be in a more readable format
    >result shape = (b, 3, 128)

3. PosEmbed(x) -> residual = x + self.W_pos[:x.shape[-2]]
    > W_pos = (3,128)     each position gets a unique, learned encoding
    > so the indexing stuff is mostly pointless. I guess it's there
     in case you used sequences of varying length or something
     because self.W_pos[:x.shape[-2]] = self.W_pos assuming l = 3
    > otherwise it's just adding the learned embeddings per position
    result shape (b, 3, 128)

4. Att(x) -> result = concat(softmax(masked_scores / sqrt(32)) * v)
    >result shape (b,3,128) or (b,3,4,32) if we exclude concat

5. Proj_Attn(x) -> result = torch.einsum('df,bqf->bqd', self.W_O, x)
    > W_0 = (128,128)
    > just a linear layer
    >result shape (b, 3, 128)

6. mid_residual = x + start_residual

7. mlp(x) ->
            x = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
            x = relu(x)
            result = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
    > relu and bias in the hidden layer
    > only the bias for the output layer
    > W_in = (512, 128); b_in (512)
    > W_out = (128, 512); b_out (128)    
    > the einsum operations allow us to do things per batch and token
    > result shape (b, 3, 128)             

8. end_residual = x + mid_residual

9. Unembed(x) -> result = (x @ self.W_U)
    > W_U = (128, 114)
    > This part is a literal matrix multiplication
    > result shape (b, 3, 114)


"""




















