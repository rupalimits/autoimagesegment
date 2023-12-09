
import pandas as pd
from training import run_training
from utilities.utilities_common import *
from image_captioning.config.core import *
from sklearn.model_selection import train_test_split

# selecting processor
if torch.cuda.is_available():

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

if __name__ == '__main__':

    # read the data
    df_data = pd.read_csv(CAPTIONS_DIR)

    # split the data into training and validation
    df_train, temp_df = train_test_split(df_data, test_size=0.2, random_state=config.lmodel_config.SEED)
    df_val, df_test = train_test_split(temp_df, test_size=0.5, random_state=config.lmodel_config.SEED)

    # train the model
    run_training(str_image_dir_path=IMAGES_DIR, df_train=df_train, df_validation=df_val)

