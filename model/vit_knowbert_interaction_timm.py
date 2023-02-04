import sys

sys.path.append('./ViT-pytorch')
sys.path.append('./kb')
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops.layers.torch import Rearrange
from models.modeling import CONFIGS, VisionTransformer
from timm.models import create_model
from torch import nn

from allennlp.common import Params
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from model.cross_attention import CrossLayer, CrossLayerWithResLink
from model.heads import generate_head
from utils.save_n_load import *
from utils.seed import *


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.concat_knowbert_last_feature = False
        self.use_reslink = False
        if self.cfg.VISION_BACKBONE == 'vit':
            if self.cfg.USE_TIMM:
                self.vision_net = create_model(
                    'vit_base_patch16_224',
                    pretrained=True,
                    num_classes=self.cfg.NUM_CLASS,
                )
            else:
                config = CONFIGS['ViT-B_16']
                self.vision_net = VisionTransformer(
                    config,
                    224,
                    zero_head=True,
                    num_classes=self.cfg.NUM_CLASS)
                self.vision_net.load_from(
                    np.load(cfg.VIT_IMAGENET_CHECKPOINT_PATH))

            self.vision_net.head = nn.Identity()

        generate_head(self)

        self.only_concat = True

        archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'

        # load model and batcher
        params = Params({'archive_file': archive_file})
        self.knowbert = ModelArchiveFromParams.from_params(params=params)
        if not self.only_concat:
            knowbert_temp = ModelArchiveFromParams.from_params(
                params=Params({'archive_file': archive_file}))
            self.bert_layer = knowbert_temp.pretrained_bert.bert.encoder.layer[
                -1]
            self.bert_pooler = knowbert_temp.pretrained_bert.bert.pooler
        self.batcher = KnowBertBatchifier(archive_file)

        self.ffn_text_bboxes = nn.Sequential(
            Rearrange("b l c -> b c l"),
            nn.Conv1d(8, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 768, kernel_size=1),
            nn.BatchNorm1d(768),
            nn.GELU(),
            Rearrange("b c l -> b l c"),
        )
        self.criterion_concat = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=cfg.vkac_dropout)
        if self.use_reslink:
            self.cross_layer = CrossLayerWithResLink()
        else:
            self.cross_layer = CrossLayer()

    def convert_kb_inputs_to_device(self, inputs, device):
        inputs['tokens']['tokens'] = inputs['tokens']['tokens'].to(device)
        inputs['segment_ids'] = inputs['segment_ids'].to(device)

        inputs['candidates']['wiki']['candidate_entity_priors'] = inputs[
            'candidates']['wiki']['candidate_entity_priors'].to(device)
        inputs['candidates']['wiki']['candidate_entities']['ids'] = inputs[
            'candidates']['wiki']['candidate_entities']['ids'].to(device)
        inputs['candidates']['wiki']['candidate_spans'] = inputs['candidates'][
            'wiki']['candidate_spans'].to(device)
        inputs['candidates']['wiki']['candidate_segment_ids'] = inputs[
            'candidates']['wiki']['candidate_segment_ids'].to(device)

        inputs['candidates']['wordnet']['candidate_entity_priors'] = inputs[
            'candidates']['wordnet']['candidate_entity_priors'].to(device)
        inputs['candidates']['wordnet']['candidate_entities']['ids'] = inputs[
            'candidates']['wordnet']['candidate_entities']['ids'].to(device)
        inputs['candidates']['wordnet']['candidate_spans'] = inputs[
            'candidates']['wordnet']['candidate_spans'].to(device)
        inputs['candidates']['wordnet']['candidate_segment_ids'] = inputs[
            'candidates']['wordnet']['candidate_segment_ids'].to(device)
        return inputs

    def vit_forward(self, img):
        if self.cfg.VISION_BACKBONE == 'vit':
            if self.cfg.USE_TIMM:
                vit_output = self.vision_net(img)
            else:
                vit_output, _ = self.vision_net(img)
        elif self.cfg.VISION_BACKBONE == 'resnet152':
            vit_output, _ = self.vision_net(img, None, None)
        return vit_output

    def attention(self, vision_f, knowledge_f, kb_f, attention_mask):
        q = vision_f[:, None]
        if self.use_reslink:
            kb_f = kb_f[:, None]
            attened_f = self.cross_layer(knowledge_f, q, kb_f,
                                         attention_mask)[:, 0]
            attened_f = self.dropout(attened_f)
        else:
            attention_mask = None
            attened_f = self.cross_layer(knowledge_f, q, attention_mask)[:, 0]
            attened_f = self.dropout(attened_f)
        return vision_f, attened_f

    def forward(self, img, texts, targets=None, text_bboxes=None):
        vit_f = self.vit_forward(img)
        kb_input = [
            batch for batch in self.batcher.iter_batches(texts, verbose=False)
        ][0]
        kb_input = self.convert_kb_inputs_to_device(kb_input, img.device)
        output = self.knowbert(**kb_input)
        kb_context, kb_f, attention_mask = output[
            'contextual_embeddings'], output['pooled_output'], None
        vit_f, attend_know_f = self.attention(vit_f, kb_context, kb_f,
                                              attention_mask)

        vit_f = self.head_vision(vit_f)
        att_f = self.head_text(attend_know_f)

        if self.concat_knowbert_last_feature:
            kb_f = self.head_text_last(kb_f)
            concat_feature = torch.cat([vit_f, att_f, kb_f], dim=-1)
        else:
            concat_feature = torch.cat([vit_f, att_f], dim=-1)
        concat_logits = self.head(concat_feature)
        if targets == None:
            return concat_logits
        else:
            loss_concat = self.criterion_concat(concat_logits, targets)
            return loss_concat


if __name__ == '__main__':
    net = Net()
