import math

import torch
from pytorch_pretrained_bert.modeling import BertLayerNorm
from torch import nn


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertIntermediate(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 hidden_act='gelu'):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act] \
            if isinstance(hidden_act, str) else hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertSelfOutput(nn.Module):

    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttention(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.1):
        super(BertCrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, query_states, attention_mask=None):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_mask = attention_scores * 0 + 1
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CrossAttention(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.self = BertCrossAttention(hidden_size, num_attention_heads,
                                       attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, attention_probs_dropout_prob)

    def forward(self, key_tensor, query_tensor, attention_mask=None):
        self_output = self.self(key_tensor, query_tensor, attention_mask)
        attention_output = self.output(self_output, query_tensor)
        return attention_output


class CrossLayer(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.1,
                 intermediate_size=3072,
                 hidden_act='gelu'):
        super(CrossLayer, self).__init__()
        self.attention = CrossAttention(hidden_size, num_attention_heads,
                                        attention_probs_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size,
                                             hidden_act)
        self.output = BertOutput(hidden_size, intermediate_size)

    def forward(self, key_tensor, query_tensor, attention_mask=None):
        attention_output = self.attention(key_tensor, query_tensor,
                                          attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CrossLayerWithResLink(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.1,
                 intermediate_size=3072,
                 hidden_act='gelu'):
        super(CrossLayerWithResLink, self).__init__()
        self.attention = CrossAttention(hidden_size, num_attention_heads,
                                        attention_probs_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size,
                                             hidden_act)
        self.output = BertOutput(hidden_size, intermediate_size)

        self.intermediate2 = BertIntermediate(hidden_size, intermediate_size,
                                              hidden_act)
        self.output2 = BertOutput(hidden_size, intermediate_size)

    def forward(self,
                key_tensor,
                query_tensor,
                other_res,
                attention_mask=None):
        attention_output = self.attention(key_tensor, query_tensor,
                                          attention_mask)
        intermediate_output = self.intermediate(attention_output + other_res)
        layer_output = self.output(intermediate_output, attention_output)

        # intermediate_output = self.intermediate2(layer_output)
        # layer_output = self.output2(intermediate_output, other_res)

        return layer_output


class TwoModalLayer(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.1,
                 intermediate_size=3072,
                 hidden_act='gelu'):
        super(TwoModalLayer, self).__init__()
        self.cross1 = CrossLayer(hidden_size, num_attention_heads,
                                 attention_probs_dropout_prob,
                                 intermediate_size, hidden_act)
        self.cross2 = CrossLayer(hidden_size, num_attention_heads,
                                 attention_probs_dropout_prob,
                                 intermediate_size, hidden_act)

    def forward(self,
                modal1,
                modal2,
                attention_mask1=None,
                attention_mask2=None):
        modal1_out = self.cross1(modal2, modal1, attention_mask2)
        modal2_out = self.cross1(modal1, modal2, attention_mask1)
        return modal1_out, modal2_out


if __name__ == '__main__':
    # bert_cross_attention = TwoModalLayer()
    # m1 = torch.randn([4,12,768])
    # m2 = torch.randn([4,24,768])
    # new_m1, new_m2 = bert_cross_attention(m1,m2)
    # print(new_m1.shape, new_m2.shape)
    # layer = CrossLayer()
    layer = CrossLayerWithResLink()
    q = torch.randn([4, 1, 768])
    k = torch.randn([4, 24, 768])
    out = layer(k, q, q)
    print(out.shape)
