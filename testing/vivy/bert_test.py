from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, register_model_architecture
from transformers import BertTokenizer, BertModel

class BERT(FairseqEncoder):
    
    def __init__(
        self,
        args,
        dictionary
    ):
        
        # Generalize attributes and methods
        super().__init__(dictionary)
        
        # Store args as instance variables
        self.args = args
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # Initialize model
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
    
    def forward(
        self, 
        src_token, 
        src_length
    ):
        
        # Get output of BERT
        output = self.model(src_token)
        
        # Return
        return output
    
@register_model('bert_test')
class BERTModel(FairseqEncoderModel):
    
    @classmethod
    def build_model(cls, args, task):
        
        bert = BERT()
        
        # Intialize BERT
        model = BERTModel()
        
        # Return model
        return model

    