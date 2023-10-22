import os
import torch

class CustomRegressionTester:
  def __init__(self, project_name, model, test_data_loader, transforms, checkpoint_file_path):
    self.project_name = project_name
    self.model = model
    self.test_data_loader = test_data_loader
    self.transforms = transforms

    self.latest_file_path = os.path.join(
      checkpoint_file_path, f"{project_name}_checkpoint_latest.pt"
    )

    print("MODEL FILE: {0}".format(self.latest_file_path))

    self.model.load_state_dict(torch.load(self.latest_file_path, map_location=torch.device('cpu')))

  def test(self):
    self.model.eval()    # Explained at 'Diverse Techniques' section

    with torch.no_grad():
      for test_batch in self.test_data_loader:
        input_test = test_batch['input']
        target_test = test_batch['target']

        if self.transforms:
          input_test = self.transforms(input_test)

        output_test = self.model(input_test)

        for output_daily, target_daily in zip(output_test, target_test):
          for date, (output, target) in enumerate(zip(output_daily, target_daily)):
            print("{0:2}: {1:6.2f} {2:6.2f} (Loss: {3:6.2f})".format(
              date, output.item(), target.item(), (target.item() - output.item())
            ))
