from datetime import datetime
import os
import torch
from torch import nn

from _01_code._99_common_utils.utils import strfdelta


class ClassificationTester:
    def __init__(self, project_name, model, test_data_loader):
        self.project_name = project_name
        self.model = model
        self.test_data_loader = test_data_loader

        self.latest_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", f"{project_name}_checkpoint_latest.pt"
        )

        torch.save(model.state_dict(), self.latest_file_path)
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

    def test(self):
        num_corrects_test = 0
        num_tested_samples = 0

        with torch.no_grad():
            for test_batch in self.test_data_loader:
                input_test, target_test = test_batch
                output_test = self.model(input_test)

                predicted_test = torch.argmax(output_test, dim=1)
                num_corrects_test += torch.sum(torch.eq(predicted_test, target_test))

                num_tested_samples += len(test_batch)

            test_accuracy = num_corrects_test / num_tested_samples

        print(f"TEST RESULTS: {test_accuracy:6.3f}")
