


from base_model import BaseModel

class DosBaseModel(BaseModel):

    def __init__(self):
        super().__init__(self)
        self.backbone = None
        self.neck = None
        self.head = None

    def train_step(self, data, n, w):

         with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)

            deep_feature = self.neck(self.backbone(data['inputs']))

            for w_i in w:
                f_loss += f(deep_feat[0], n, w_i)
            print(f_loss)


        parsed_losses, log_vars = self.parse_losses(losses)j
        optim_wrapper.update_params(parsed_losses)
        return log_vars
