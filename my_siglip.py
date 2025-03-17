from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768, # embeding size
        intermediate_size=3072, # linear layer size(for mlp use)
        num_hidden_layers=12, # layer number
        num_attention_heads=12,
        num_channels=3,
        image_size=224, # image size(224*224的图像)
        patch_size=16, # patch size(每张图被分成16*16的小块)
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None, #即最终的图像token数量（一个图像用几个向量表示）
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        b_s,channel,h,w = pixel_values.shape # batch_size,channel,height,wide
        # batch_size,channel,height,wide -> batch_size, embed_dim, height/patch_size, wide/patch_size
        conv_tensor = self.patch_embedding(pixel_values) 
        # batch_size, embed_dim, height/patch_size, wide/patch_size -> batch_size, embed_dim, num_positions
        conv_tensor = conv_tensor.flatten(2) 
        # batch_size, embed_dim, num_positions -> batch_size, num_positions, embed_dim
        conv_tensor = conv_tensor.transpose(1,2)
        conv_tensor = conv_tensor + self.position_embedding(self.position_ids)
        return conv_tensor # embedings

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        # 这里除以num_heads，是因为要把embed_dim分成num_heads份，每份的维度是head_dim，这样可以并行计算了
        self.head_dim = self.embed_dim // self.num_heads 
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, num_patches, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_heads, num_patches, head_Dim]
        query_states = query_states.view(batch_size,num_patches,self.num_heads,self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size,num_patches,self.num_heads,self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size,num_patches,self.num_heads,self.head_dim).transpose(1,2)
        
        # att_score: [Batch_Size, Num_heads, num_patches, num_patches] also called att_weight
        att_score = torch.matmul(query_states,key_states.transpose(2,3)) * self.scale
        # 对于语言模型，需要mask的话，在这里实现mask，最简单的实现就是矩阵对角线以上的元素全部置为负无穷
        att_score = torch.softmax(att_score,dim=-1)
        att_score = nn.functional.dropout(att_score, p=self.dropout, training=self.training)

        # att_output: [Batch_Size, Num_heads, num_patches, head_Dim]
        att_output = torch.matmul(att_score,value_states)
        if att_output.size() != (batch_size, self.num_heads, num_patches, self.head_dim):
            raise ValueError(
                f"`att_output` should be of size {(batch_size, self.num_heads, num_patches, self.head_dim)}, but is"
                f" {att_output.size()}"
            )
        att_output = att_output.transpose(1,2).contiguous() # 似乎用reshape，contiguous不加也行
        att_output = att_output.reshape(batch_size,num_patches,self.head_dim)
        att_output = self.out_proj(att_output) # 这一步是为了混合多个注意力头的结果
        #att_output: [Batch_Size, Num_Patches, Embed_Dim]
        return att_output # (有时也会返回注意力分数，但是这里我们没用到就不返回了)

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        # self_attn不改变输入输出向量的形状，只是让每个向量都拥有上下文的信息
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    def forward(self,embedings : torch.Tensor) -> torch.Tensor:
        output = embedings
        for layer in self.layers():
            output = layer(output)
        return output

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 