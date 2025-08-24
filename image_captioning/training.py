
from utilities.utilities_common import *
from config.core import *
from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments
from transformers import default_data_collator, VisionEncoderDecoderModel


def run_training(str_image_dir_path, df_train, df_validation):

    # transform the training and validation dataframes
    train_dataset = ImgDataset(df_train, root_dir=str_image_dir_path, tokenizer=tokenizer, feature_extractor=feature_extractor, transform=img_transforms)
    validation_dataset = ImgDataset(df_validation, root_dir=str_image_dir_path, tokenizer=tokenizer, feature_extractor=feature_extractor, transform=img_transforms)

    # initialize the model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.lmodel_config.ENCODER, config.lmodel_config.DECODER)
    # set model config parameters
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.max_length = config.lmodel_config.MAX_LEN
    model.config.early_stopping = config.lmodel_config.EARLY_STOPPING
    model.config.no_repeat_ngram_size = config.lmodel_config.NGRAM_SIZE
    model.config.length_penalty = config.lmodel_config.LEN_PENALTY
    model.config.num_beams = config.lmodel_config.NUM_BEAMS

    # define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='VIT_large_gpt2',
        per_device_train_batch_size=config.lmodel_config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.lmodel_config.VAL_BATCH_SIZE,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=config.lmodel_config.NUM_LOGGING_STEPS,
        save_steps=2 * config.lmodel_config.NUM_LOGGING_STEPS,
        warmup_steps=config.lmodel_config.NUM_LOGGING_STEPS,
        learning_rate=config.lmodel_config.LR,
        max_steps=1500, # delete for full training
        num_train_epochs=config.lmodel_config.EPOCHS,  # TRAIN_EPOCHS
        overwrite_output_dir=True,
        save_total_limit=1,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        tokenizer=feature_extractor,
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    # save the trained model
    trainer.save_model(IMAGES_DIR / 'VIT_large_gpt2')
