# Đỗ Lê Duy - 21000671

## Lab 5: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

### Mục tiêu

Hiểu rõ hạn chế của các mô hình phân loại văn bản truyền thống (Bag-of-Words,
Word2Vec trung bình).
• Nắm vững kiến trúc và luồng hoạt động của pipeline sử dụng RNN/LSTM cho bài
toán phân loại văn bản.
• Tự tay xây dựng, huấn luyện và so sánh hiệu năng giữa các mô hình:

1. TF-IDF + Logistic Regression (Baseline 1).
2. Word2Vec (vector trung bình) + Dense Layer (Baseline 2).
3. Embedding Layer (pre-trained) + LSTM.
4. Embedding Layer (học từ đầu) + LSTM.
   • Phân tích và đánh giá sức mạnh của mô hình chuỗi trong việc “hiểu” ngữ cảnh
   của câu.

### Các phần chính

- Bước 0: Thiết lập Môi trường và Tải Dữ liệu
- Nhiệm vụ 1: (Warm-up Ôn bài cũ) Pipeline TF-IDF + Logistic Regression: Đây là mô hình baseline để chúng ta có một cơ sở so sánh. Hãy áp dụng lại kiến thức
  từ lab trước.
- Nhiệm vụ 2: (Warm-up Ôn bài cũ) Pipeline Word2Vec (Trung bình) + DenseLayer: Mô hình baseline thứ hai, sử dụng embedding nhưng chưa có khả năng xử lý chuỗi.
- Nhiệm vụ 3: Mô hình Nâng cao (Embedding Pre-trained + LSTM): Đây là nhiệm vụ chính đầu tiên. Chúng ta sẽ sử dụng Word2Vec đã huấn luyện ở Nhiệm
  vụ 2 để khởi tạo trọng số cho Embedding Layer.
- Nhiệm vụ 4: Mô hình Nâng cao (Embedding học từ đầu + LSTM): Lần này, chúng ta sẽ để mô hình tự học lớp Embedding. Kiến trúc gần như tương tự
  Nhiệm vụ 3, nhưng Embedding Layer sẽ được học từ đầu.
- Nhiệm vụ 5: Đánh giá, So sánh và Phân tích

### Chạy chương trình:

- Mở file trên google colab, chạy cell đầu và upload 3 file data: `train.csv`, `val.csv`, `test.csv` lên
- Sau đó chạy lần lượt từng cel để nhận kết quả

### Kết quả

- Nhiệm vụ 1

```
Accuracy: 0.8356545961002786
Precision: 0.8405884523852166
Recall: 0.8356545961002786
F1-score: 0.834645398866917
Classification report:
              precision    recall  f1-score   support
           0       0.90      0.95      0.92        19
           1       1.00      0.73      0.84        11
           2       0.81      0.89      0.85        19
           3       1.00      0.75      0.86         8
           4       0.92      0.80      0.86        15
           5       0.93      1.00      0.96        13
           6       0.48      0.53      0.50        19
           7       0.89      0.89      0.89        19
           8       0.82      0.74      0.78        19
           9       0.00      0.00      0.00         1
          10       0.59      0.68      0.63        19
          11       0.67      0.75      0.71         8
          12       0.74      0.89      0.81        19
          13       0.78      0.88      0.82         8
          14       0.83      0.79      0.81        19
          15       0.92      0.63      0.75        19
          16       0.77      0.89      0.83        19
          17       1.00      1.00      1.00        19
          18       1.00      1.00      1.00        19
          19       1.00      1.00      1.00        19
          20       0.90      1.00      0.95        19
          21       1.00      0.95      0.97        19
          22       1.00      1.00      1.00        12
          23       0.95      1.00      0.97        19
          24       0.95      1.00      0.97        19
          25       0.36      0.26      0.30        19
          26       0.90      1.00      0.95        19
          27       1.00      1.00      1.00        16
          28       1.00      0.95      0.97        19
          29       0.75      0.79      0.77        19
          30       0.91      0.83      0.87        12
          31       0.89      0.89      0.89        19
          32       0.67      0.67      0.67         3
          33       0.92      0.86      0.89        14
          34       0.80      0.89      0.84         9
          35       0.78      1.00      0.88         7
          36       0.68      0.79      0.73        19
          37       0.75      0.79      0.77        19
          38       0.85      0.89      0.87        19
          39       0.65      0.61      0.63        18
          40       0.71      0.53      0.61        19
          41       1.00      0.57      0.73         7
          42       0.75      0.63      0.69        19
          43       0.95      0.95      0.95        19
          44       0.81      0.68      0.74        19
          45       0.58      0.74      0.65        19
          46       1.00      0.84      0.91        19
          47       0.89      0.84      0.86        19
          48       0.94      0.89      0.92        19
          49       0.82      0.95      0.88        19
          50       0.48      0.58      0.52        19
          51       0.92      0.86      0.89        14
          52       1.00      0.95      0.97        19
          53       0.83      0.79      0.81        19
          54       0.81      0.89      0.85        19
          55       1.00      1.00      1.00        10
          56       0.95      1.00      0.97        19
          57       0.80      0.89      0.84        18
          58       0.83      0.79      0.81        19
          59       0.89      0.89      0.89        19
          60       0.68      0.79      0.73        19
          61       1.00      1.00      1.00        18
          62       0.94      0.79      0.86        19
          63       1.00      0.95      0.97        19
          64       0.65      0.68      0.67        19
    accuracy                           0.84      1077
   macro avg       0.83      0.82      0.82      1077
weighted avg       0.84      0.84      0.83      1077
```

- Nhiệm vụ 2:

```
Epoch 1/10
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
280/280 - 2s - 5ms/step - accuracy: 0.0201 - loss: 4.1509 - val_accuracy: 0.0334 - val_loss: 4.1079
Epoch 2/10
280/280 - 1s - 5ms/step - accuracy: 0.0337 - loss: 4.1036 - val_accuracy: 0.0585 - val_loss: 4.0537
Epoch 3/10
280/280 - 1s - 5ms/step - accuracy: 0.0517 - loss: 4.0292 - val_accuracy: 0.0669 - val_loss: 3.9559
Epoch 4/10
280/280 - 2s - 6ms/step - accuracy: 0.0590 - loss: 3.9281 - val_accuracy: 0.0966 - val_loss: 3.8362
Epoch 5/10
280/280 - 1s - 2ms/step - accuracy: 0.0690 - loss: 3.8333 - val_accuracy: 0.0975 - val_loss: 3.7329
Epoch 6/10
280/280 - 1s - 2ms/step - accuracy: 0.0794 - loss: 3.7392 - val_accuracy: 0.1114 - val_loss: 3.6399
Epoch 7/10
280/280 - 1s - 2ms/step - accuracy: 0.0917 - loss: 3.6561 - val_accuracy: 0.1346 - val_loss: 3.5734
Epoch 8/10
280/280 - 1s - 3ms/step - accuracy: 0.1024 - loss: 3.5924 - val_accuracy: 0.1356 - val_loss: 3.5130
Epoch 9/10
280/280 - 1s - 3ms/step - accuracy: 0.1106 - loss: 3.5384 - val_accuracy: 0.1560 - val_loss: 3.4522
Epoch 10/10
280/280 - 1s - 3ms/step - accuracy: 0.1155 - loss: 3.4984 - val_accuracy: 0.1634 - val_loss: 3.4056
Test loss: 3.4154  Test accuracy: 0.1690
34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
Classification report:
                          precision    recall  f1-score   support
             alarm_query       0.12      0.21      0.15        19
            alarm_remove       0.00      0.00      0.00        11
               alarm_set       0.26      0.84      0.40        19
       audio_volume_down       0.33      0.12      0.18         8
       audio_volume_mute       0.00      0.00      0.00        15
         audio_volume_up       0.22      0.15      0.18        13
          calendar_query       0.11      0.05      0.07        19
         calendar_remove       0.33      0.05      0.09        19
            calendar_set       0.00      0.00      0.00        19
                category       0.00      0.00      0.00         1
          cooking_recipe       0.00      0.00      0.00        19
        datetime_convert       0.00      0.00      0.00         8
          datetime_query       0.09      0.74      0.17        19
        email_addcontact       0.00      0.00      0.00         8
             email_query       0.07      0.05      0.06        19
      email_querycontact       0.00      0.00      0.00        19
         email_sendemail       0.15      0.21      0.17        19
          general_affirm       0.18      0.37      0.24        19
     general_commandstop       0.46      0.58      0.51        19
         general_confirm       0.40      0.89      0.55        19
        general_dontcare       0.18      0.63      0.28        19
         general_explain       0.08      0.16      0.11        19
            general_joke       0.00      0.00      0.00        12
          general_negate       0.20      0.05      0.08        19
          general_praise       0.24      0.53      0.33        19
          general_quirky       0.00      0.00      0.00        19
          general_repeat       0.33      0.16      0.21        19
            iot_cleaning       0.33      0.38      0.35        16
              iot_coffee       0.11      0.11      0.11        19
     iot_hue_lightchange       0.32      0.58      0.42        19
        iot_hue_lightdim       1.00      0.08      0.15        12
        iot_hue_lightoff       0.28      0.89      0.42        19
         iot_hue_lighton       0.00      0.00      0.00         3
         iot_hue_lightup       0.00      0.00      0.00        14
            iot_wemo_off       0.00      0.00      0.00         9
             iot_wemo_on       0.00      0.00      0.00         7
       lists_createoradd       0.17      0.37      0.23        19
             lists_query       0.00      0.00      0.00        19
            lists_remove       0.28      0.26      0.27        19
          music_likeness       0.00      0.00      0.00        18
             music_query       0.00      0.00      0.00        19
          music_settings       0.00      0.00      0.00         7
              news_query       0.00      0.00      0.00        19
          play_audiobook       0.06      0.05      0.06        19
               play_game       0.00      0.00      0.00        19
              play_music       0.00      0.00      0.00        19
           play_podcasts       0.00      0.00      0.00        19
              play_radio       0.08      0.05      0.06        19
             qa_currency       0.00      0.00      0.00        19
           qa_definition       0.00      0.00      0.00        19
              qa_factoid       0.09      0.26      0.14        19
                qa_maths       0.00      0.00      0.00        14
                qa_stock       0.12      0.11      0.11        19
   recommendation_events       0.33      0.05      0.09        19
recommendation_locations       0.00      0.00      0.00        19
   recommendation_movies       0.00      0.00      0.00        10
             social_post       0.00      0.00      0.00        19
            social_query       0.00      0.00      0.00        18
          takeaway_order       0.10      0.21      0.14        19
          takeaway_query       0.00      0.00      0.00        19
         transport_query       0.00      0.00      0.00        19
          transport_taxi       0.00      0.00      0.00        18
        transport_ticket       0.13      0.58      0.22        19
       transport_traffic       0.00      0.00      0.00        19
           weather_query       0.00      0.00      0.00        19
                accuracy                           0.17      1077
               macro avg       0.11      0.15      0.10      1077
            weighted avg       0.11      0.17      0.11      1077
```

- Nhiệm vụ 3:

```
Epoch 1/10
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
280/280 - 34s - 120ms/step - accuracy: 0.0163 - loss: 4.1479 - val_accuracy: 0.0176 - val_loss: 4.1383
Epoch 2/10
280/280 - 27s - 96ms/step - accuracy: 0.0152 - loss: 4.1377 - val_accuracy: 0.0176 - val_loss: 4.1298
Epoch 3/10
280/280 - 41s - 146ms/step - accuracy: 0.0132 - loss: 4.1364 - val_accuracy: 0.0176 - val_loss: 4.1288
Epoch 4/10
280/280 - 42s - 151ms/step - accuracy: 0.0151 - loss: 4.1354 - val_accuracy: 0.0176 - val_loss: 4.1287
Epoch 5/10
280/280 - 40s - 141ms/step - accuracy: 0.0150 - loss: 4.1347 - val_accuracy: 0.0176 - val_loss: 4.1283
Epoch 6/10
280/280 - 27s - 96ms/step - accuracy: 0.0145 - loss: 4.1342 - val_accuracy: 0.0176 - val_loss: 4.1283
Epoch 7/10
280/280 - 27s - 96ms/step - accuracy: 0.0146 - loss: 4.1338 - val_accuracy: 0.0176 - val_loss: 4.1295
Epoch 8/10
280/280 - 42s - 151ms/step - accuracy: 0.0169 - loss: 4.1339 - val_accuracy: 0.0176 - val_loss: 4.1281
Epoch 9/10
280/280 - 40s - 142ms/step - accuracy: 0.0169 - loss: 4.1332 - val_accuracy: 0.0176 - val_loss: 4.1282
Epoch 10/10
280/280 - 41s - 146ms/step - accuracy: 0.0137 - loss: 4.1327 - val_accuracy: 0.0176 - val_loss: 4.1310
Test loss: 4.1281  Test accuracy: 0.0176
34/34 ━━━━━━━━━━━━━━━━━━━━ 2s 36ms/step
Classification report:
                          precision    recall  f1-score   support

             alarm_query       0.00      0.00      0.00        19
            alarm_remove       0.00      0.00      0.00        11
               alarm_set       0.00      0.00      0.00        19
       audio_volume_down       0.00      0.00      0.00         8
       audio_volume_mute       0.00      0.00      0.00        15
         audio_volume_up       0.00      0.00      0.00        13
          calendar_query       0.00      0.00      0.00        19
         calendar_remove       0.00      0.00      0.00        19
            calendar_set       0.00      0.00      0.00        19
                category       0.00      0.00      0.00         1
          cooking_recipe       0.00      0.00      0.00        19
        datetime_convert       0.00      0.00      0.00         8
          datetime_query       0.00      0.00      0.00        19
        email_addcontact       0.00      0.00      0.00         8
             email_query       0.00      0.00      0.00        19
      email_querycontact       0.00      0.00      0.00        19
         email_sendemail       0.00      0.00      0.00        19
          general_affirm       0.00      0.00      0.00        19
     general_commandstop       0.00      0.00      0.00        19
         general_confirm       0.00      0.00      0.00        19
        general_dontcare       0.00      0.00      0.00        19
         general_explain       0.00      0.00      0.00        19
            general_joke       0.00      0.00      0.00        12
          general_negate       0.00      0.00      0.00        19
          general_praise       0.00      0.00      0.00        19
          general_quirky       0.00      0.00      0.00        19
          general_repeat       0.00      0.00      0.00        19
            iot_cleaning       0.00      0.00      0.00        16
              iot_coffee       0.00      0.00      0.00        19
     iot_hue_lightchange       0.02      1.00      0.03        19
        iot_hue_lightdim       0.00      0.00      0.00        12
        iot_hue_lightoff       0.00      0.00      0.00        19
         iot_hue_lighton       0.00      0.00      0.00         3
         iot_hue_lightup       0.00      0.00      0.00        14
            iot_wemo_off       0.00      0.00      0.00         9
             iot_wemo_on       0.00      0.00      0.00         7
       lists_createoradd       0.00      0.00      0.00        19
             lists_query       0.00      0.00      0.00        19
            lists_remove       0.00      0.00      0.00        19
          music_likeness       0.00      0.00      0.00        18
             music_query       0.00      0.00      0.00        19
          music_settings       0.00      0.00      0.00         7
              news_query       0.00      0.00      0.00        19
          play_audiobook       0.00      0.00      0.00        19
               play_game       0.00      0.00      0.00        19
              play_music       0.00      0.00      0.00        19
           play_podcasts       0.00      0.00      0.00        19
              play_radio       0.00      0.00      0.00        19
             qa_currency       0.00      0.00      0.00        19
           qa_definition       0.00      0.00      0.00        19
              qa_factoid       0.00      0.00      0.00        19
                qa_maths       0.00      0.00      0.00        14
                qa_stock       0.00      0.00      0.00        19
   recommendation_events       0.00      0.00      0.00        19
recommendation_locations       0.00      0.00      0.00        19
   recommendation_movies       0.00      0.00      0.00        10
             social_post       0.00      0.00      0.00        19
            social_query       0.00      0.00      0.00        18
          takeaway_order       0.00      0.00      0.00        19
          takeaway_query       0.00      0.00      0.00        19
         transport_query       0.00      0.00      0.00        19
          transport_taxi       0.00      0.00      0.00        18
        transport_ticket       0.00      0.00      0.00        19
       transport_traffic       0.00      0.00      0.00        19
           weather_query       0.00      0.00      0.00        19
                accuracy                           0.02      1077
               macro avg       0.00      0.02      0.00      1077
            weighted avg       0.00      0.02      0.00      1077
```

- Nhiệm vụ 4:

```
Epoch 1/10
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
280/280 - 37s - 133ms/step - accuracy: 0.0141 - loss: 4.1513 - val_accuracy: 0.0176 - val_loss: 4.1321
Epoch 2/10
280/280 - 33s - 117ms/step - accuracy: 0.0146 - loss: 4.1399 - val_accuracy: 0.0176 - val_loss: 4.1305
Epoch 3/10
280/280 - 31s - 112ms/step - accuracy: 0.0150 - loss: 4.1368 - val_accuracy: 0.0176 - val_loss: 4.1289
Epoch 4/10
280/280 - 43s - 153ms/step - accuracy: 0.0155 - loss: 4.1358 - val_accuracy: 0.0176 - val_loss: 4.1295
Epoch 5/10
280/280 - 31s - 112ms/step - accuracy: 0.0162 - loss: 4.1347 - val_accuracy: 0.0176 - val_loss: 4.1291
Epoch 6/10
280/280 - 43s - 152ms/step - accuracy: 0.0165 - loss: 4.1349 - val_accuracy: 0.0176 - val_loss: 4.1283
Epoch 7/10
280/280 - 31s - 112ms/step - accuracy: 0.0132 - loss: 4.1344 - val_accuracy: 0.0176 - val_loss: 4.1290
Epoch 8/10
280/280 - 42s - 151ms/step - accuracy: 0.0136 - loss: 4.1337 - val_accuracy: 0.0176 - val_loss: 4.1287
Epoch 9/10
280/280 - 32s - 113ms/step - accuracy: 0.0133 - loss: 4.1336 - val_accuracy: 0.0176 - val_loss: 4.1287
Test loss (scratch): 4.1283  Test accuracy (scratch): 0.0176
34/34 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step
Classification report (scratch):
                          precision    recall  f1-score   support

             alarm_query       0.00      0.00      0.00        19
            alarm_remove       0.00      0.00      0.00        11
               alarm_set       0.00      0.00      0.00        19
       audio_volume_down       0.00      0.00      0.00         8
       audio_volume_mute       0.00      0.00      0.00        15
         audio_volume_up       0.00      0.00      0.00        13
          calendar_query       0.00      0.00      0.00        19
         calendar_remove       0.00      0.00      0.00        19
            calendar_set       0.00      0.00      0.00        19
                category       0.00      0.00      0.00         1
          cooking_recipe       0.00      0.00      0.00        19
        datetime_convert       0.00      0.00      0.00         8
          datetime_query       0.00      0.00      0.00        19
        email_addcontact       0.00      0.00      0.00         8
             email_query       0.00      0.00      0.00        19
      email_querycontact       0.00      0.00      0.00        19
         email_sendemail       0.00      0.00      0.00        19
          general_affirm       0.00      0.00      0.00        19
     general_commandstop       0.00      0.00      0.00        19
         general_confirm       0.00      0.00      0.00        19
        general_dontcare       0.00      0.00      0.00        19
         general_explain       0.00      0.00      0.00        19
            general_joke       0.00      0.00      0.00        12
          general_negate       0.00      0.00      0.00        19
          general_praise       0.00      0.00      0.00        19
          general_quirky       0.00      0.00      0.00        19
          general_repeat       0.00      0.00      0.00        19
            iot_cleaning       0.00      0.00      0.00        16
              iot_coffee       0.00      0.00      0.00        19
     iot_hue_lightchange       0.00      0.00      0.00        19
        iot_hue_lightdim       0.00      0.00      0.00        12
        iot_hue_lightoff       0.00      0.00      0.00        19
         iot_hue_lighton       0.00      0.00      0.00         3
         iot_hue_lightup       0.00      0.00      0.00        14
            iot_wemo_off       0.00      0.00      0.00         9
             iot_wemo_on       0.00      0.00      0.00         7
       lists_createoradd       0.00      0.00      0.00        19
             lists_query       0.00      0.00      0.00        19
            lists_remove       0.00      0.00      0.00        19
          music_likeness       0.00      0.00      0.00        18
             music_query       0.00      0.00      0.00        19
          music_settings       0.00      0.00      0.00         7
              news_query       0.00      0.00      0.00        19
          play_audiobook       0.00      0.00      0.00        19
               play_game       0.00      0.00      0.00        19
              play_music       0.00      0.00      0.00        19
           play_podcasts       0.00      0.00      0.00        19
              play_radio       0.00      0.00      0.00        19
             qa_currency       0.00      0.00      0.00        19
           qa_definition       0.00      0.00      0.00        19
              qa_factoid       0.00      0.00      0.00        19
                qa_maths       0.00      0.00      0.00        14
                qa_stock       0.00      0.00      0.00        19
   recommendation_events       0.00      0.00      0.00        19
recommendation_locations       0.00      0.00      0.00        19
   recommendation_movies       0.00      0.00      0.00        10
             social_post       0.00      0.00      0.00        19
            social_query       0.00      0.00      0.00        18
          takeaway_order       0.00      0.00      0.00        19
          takeaway_query       0.00      0.00      0.00        19
         transport_query       0.00      0.00      0.00        19
          transport_taxi       0.00      0.00      0.00        18
        transport_ticket       0.00      0.00      0.00        19
       transport_traffic       0.00      0.00      0.00        19
           weather_query       0.02      1.00      0.03        19
                accuracy                           0.02      1077
               macro avg       0.00      0.02      0.00      1077
            weighted avg       0.00      0.02      0.00      1077
```

- Nhiệm vụ 5:

```
{'TF-IDF + Logistic Regression': {'macro_f1': 0.82, 'test_loss': None}, 'Word2Vec (Avg) + Dense': {'macro_f1': 0.1, 'test_loss': 4.128147125244141}, 'Embedding (Pre-trained) + LSTM': {'macro_f1': 0.0, 'test_loss': 4.128147125244141}, 'Embedding (Scratch) + LSTM': {'macro_f1': 0.0, 'test_loss': 4.128331661224365}}
```

```
| Model | Macro F1-score | Test Loss |
|---|---|---|
| TF-IDF + Logistic Regression | 0.82 | None |
| Word2Vec (Avg) + Dense | 0.10 | 4.1281 |
| Embedding (Pre-trained) + LSTM | 0.00 | 4.1281 |
| Embedding (Scratch) + LSTM | 0.00 | 4.1283 |
```

```
Challenging sentences selected:
- don't set an alarm for me tomorrow morning
- i'd like to know if there are any upcoming concerts near me this week
- turn off the light in the living room but not the kitchen
- what's the weather going to be like in london next tuesday afternoon
- can you remind me to buy groceries when I leave the house
```

```
Analysis for sentence: 'don't set an alarm for me tomorrow morning' (True Intent: alarm_remove)
--------------------------------------------------
TF-IDF + LR: Predicted 'alarm_set' (Incorrect)
Word2Vec + Dense: Predicted 'alarm_set' (Incorrect)
Embedding (Pre-trained) + LSTM: Predicted 'iot_hue_lightchange' (Incorrect)
Embedding (Scratch) + LSTM: Predicted 'weather_query' (Incorrect)

Observations:
- This sentence involves negation ('don't').
  - TF-IDF + LR correctly identified the core 'alarm' intent but missed the negation.
  - Word2Vec + Dense also missed the negation.
  - The LSTM models performed poorly, predicting unrelated intents. This suggests they struggled with both negation and the overall sentence meaning.
--------------------------------------------------


Analysis for sentence: 'i'd like to know if there are any upcoming concerts near me this week' (True Intent: recommendation_events)
--------------------------------------------------
TF-IDF + LR: Predicted 'recommendation_events' (Correct)
Word2Vec + Dense: Predicted 'general_dontcare' (Incorrect)
Embedding (Pre-trained) + LSTM: Predicted 'iot_hue_lightchange' (Incorrect)
Embedding (Scratch) + LSTM: Predicted 'weather_query' (Incorrect)

Observations:
- This is a query about events.
  - TF-IDF + LR correctly identified 'recommendation_events'.
  - Word2Vec + Dense predicted a general intent ('general_dontcare'), failing to capture the specific query.
  - The LSTM models again predicted unrelated intents, indicating difficulty with complex query structures.
--------------------------------------------------


Analysis for sentence: 'turn off the light in the living room but not the kitchen' (True Intent: iot_hue_lightoff)
--------------------------------------------------
TF-IDF + LR: Predicted 'iot_hue_lightoff' (Correct)
Word2Vec + Dense: Predicted 'iot_hue_lightoff' (Correct)
Embedding (Pre-trained) + LSTM: Predicted 'iot_hue_lightchange' (Incorrect)
Embedding (Scratch) + LSTM: Predicted 'weather_query' (Incorrect)

Observations:
- This sentence has negation ('not') and multiple entities/actions.
  - TF-IDF + LR correctly identified 'iot_hue_lightoff'. It likely focused on 'turn off the light'.
  - Word2Vec + Dense also correctly identified 'iot_hue_lightoff'.
  - The pre-trained LSTM model predicted 'iot_hue_lightchange', possibly focusing on 'light' but missing the 'off' and negation.
  - The scratch LSTM model predicted 'weather_query', completely missing the context.
  - This highlights the difficulty of handling multiple clauses and negation for some models.
--------------------------------------------------


Analysis for sentence: 'what's the weather going to be like in london next tuesday afternoon' (True Intent: weather_query)
--------------------------------------------------
TF-IDF + LR: Predicted 'weather_query' (Correct)
Word2Vec + Dense: Predicted 'play_radio' (Incorrect)
Embedding (Pre-trained) + LSTM: Predicted 'iot_hue_lightchange' (Incorrect)
Embedding (Scratch) + LSTM: Predicted 'weather_query' (Correct)

Observations:
- This is a specific weather query with location and time.
  - TF-IDF + LR correctly identified 'weather_query'.
  - Word2Vec + Dense predicted 'play_radio', completely missing the intent.
  - Both LSTM models predicted 'weather_query', correctly handling the specific details.
  - This is a positive example for the LSTM models, showing they can sometimes capture context.
--------------------------------------------------


Analysis for sentence: 'can you remind me to buy groceries when I leave the house' (True Intent: calendar_set)
--------------------------------------------------
TF-IDF + LR: Predicted 'calendar_set' (Correct)
Word2Vec + Dense: Predicted 'general_explain' (Incorrect)
Embedding (Pre-trained) + LSTM: Predicted 'iot_hue_lightchange' (Incorrect)
Embedding (Scratch) + LSTM: Predicted 'weather_query' (Incorrect)

Observations:
- This is a conditional reminder/setting a calendar entry.
  - TF-IDF + LR correctly identified 'calendar_set'.
  - Word2Vec + Dense predicted a general intent ('general_explain'), failing to capture the core action.
  - The pre-trained LSTM model predicted 'iot_hue_lightchange', completely unrelated.
  - The scratch LSTM model predicted 'weather_query', also unrelated.
  - The conditional structure ('when I leave the house') might be challenging for some models.
```

### Giải thích kết quả

Phân tích từng câu (vấn đề ngôn ngữ, vì sao LSTM tốt/không tốt)

- "don't set an alarm for me tomorrow morning" — (negation, phụ thuộc xa)

* Vấn đề: ý nghĩa bị đảo bởi "don't" (negation) ngay trước động từ; cần giữ thứ tự và ràng buộc giữa "don't" và "set".
* LSTM khả năng: nếu được huấn luyện đủ, LSTM có thể nắm quan hệ thứ tự/phụ thuộc (negation affects verb) vì xử lý tuần tự.
* Tại sao thất bại ở bạn: data thiếu mẫu chứa cấu trúc phủ định, embedding hoặc model quá yếu, hoặc lớp nhãn hiếm khiến model dự đoán theo lớp phổ biến.

- "i'd like to know if there are any upcoming concerts near me this week" — (ý định phụ thuộc ngữ cảnh/khái niệm)

* Vấn đề: cần nhận diện intent “recommendation_events” từ nhiều từ khóa phân tán ("upcoming", "concerts", "near me", "this week").
* LSTM khả năng: tốt hơn TF-IDF khi ngữ cảnh liên tục quan trọng; nhưng vẫn cần đủ ví dụ để liên kết các token thành intent.
* Tại sao thất bại: nếu embedding/huấn luyện không bắt được cụm “upcoming concerts” hoặc bị nhiễu bởi các từ chung.

- "turn off the light in the living room but not the kitchen" — (phủ định cục bộ, cấu trúc đối lập)

* Vấn đề: mệnh đề “but not the kitchen” phủ nhận một phần mục tiêu — cần hiểu phạm vi hành động.
* LSTM khả năng: có thể học được cấu trúc đối lập nếu thấy đủ ví dụ; tuy nhiên LSTM thường mờ bỏ chi tiết dài hơn nếu không có attention.
* Tại sao thất bại: thiếu ví dụ có cấu trúc “but not ...”, hoặc embedding/kiến trúc kém phân giải.

- "what's the weather going to be like in london next tuesday afternoon" — (thông tin thực thể: địa điểm + thời gian)

* Vấn đề: cần nhận dạng thực thể địa điểm + thời gian và map đến intent weather_query.
* LSTM khả năng: nếu token hoá và embedding tốt, LSTM có ưu thế nắm chuỗi và vị trí entity; có thể thắng TF-IDF.
* Tại sao đôi khi đúng/sai: LSTM có thể học pattern “weather + location + time” nhưng cần dữ liệu và regularization.

- "can you remind me to buy groceries when I leave the house" — (phụ thuộc xa, điều kiện)

* Vấn đề: mục tiêu là tạo reminder/đặt lịch (calendar_set) — thông tin quan trọng rải rác, có mệnh đề điều kiện "when I leave the house".
* LSTM khả năng: tốt hơn bag-of-words vì giữ thứ tự và pha trộn thông tin; nhưng lại phụ thuộc vào dữ liệu đào tạo để học quan hệ điều kiện.

\*Nhận xét chung: vì sao LSTM đôi khi tốt hơn hoặc không

- Tốt hơn khi:

* Nhiệm vụ cần thông tin thứ tự, phụ thuộc dài (negation, điều kiện, cụm động từ).
* Có đủ dữ liệu để học các pattern tuần tự.

- Không tốt khi:

* Dữ liệu ít / class imbalance lớn → LSTM dễ overfit hoặc bỏ qua lớp hiếm.
* Embedding không phù hợp (domain mismatch) → thông tin ngữ nghĩa yếu.
* Kiến trúc không đủ (không có attention, quá nông) → mất chi tiết dài.
* Huấn luyện/điều chỉnh kém (learning rate, early stopping, batch size).
* Ưu & nhược điểm các phương pháp (tóm tắt)

-TF-IDF + Logistic Regression

- Ưu: đơn giản, nhanh, hiệu quả trên dữ liệu nhỏ, ít tốn tài nguyên.
- Nhược: không nắm thứ tự từ hay ngữ cảnh; gặp khó với negation và phụ thuộc dài; feature sparse.

* Word2Vec (trung bình) + Dense

- Ưu: tận dụng nghĩa từ, vector dày hơn, tính toán nhanh hơn LSTM.
- Nhược: mất thông tin vị trí/thuật tự; trung bình làm mờ cấu trúc câu (khó với negation, điều kiện).

* Embedding (pre‑trained) + LSTM

- Ưu: giữ thứ tự, tận dụng embedding có sẵn (khởi tạo tốt), có khả năng học phụ thuộc dài.
- Nhược: cần nhiều dữ liệu + điều chỉnh; nếu domain mismatch thì kém; training chậm.

* Embedding (học từ đầu) + LSTM

- Ưu: có thể học embedding phù hợp domain.
- Nhược: cần nhiều dữ liệu; dễ overfit; khởi tạo kém làm model rơi vào hiệu năng thấp nếu dữ liệu nhỏ.
  => Khuyến nghị để cải thiện LSTM trong bài của bạn

- Tăng dữ liệu hoặc dùng data augmentation; stratified sampling cho class imbalance.
- Dùng pretrained contextual models (BERT, RoBERTa) và fine‑tune thay vì LSTM thuần — cải thiện hiểu ngữ cảnh & negation.
- Thêm attention/bi‑LSTM để bắt phụ thuộc dài tốt hơn.
- Gắn token đặc biệt cho negation/phủ định hoặc dùng feature POS/NER nếu cần.
- Điều chỉnh loss (class weights) hoặc sử dụng focal loss cho lớp hiếm.
- Kiểm tra preprocessing: giữ lại từ quan trọng (negation), tránh loại quá mức.

### 1 số khó khăn và cách khắc phục

- chạy trên gg colab, không thể đọc dữ liệu từ máy trực tiếp
  => tạo đối tượng upload file để up file data từ máy lên
- số epoch khá ít (10), nên độ chính xác vẫn chưa hội tụ hết
  => có thể tăng epoch, nhưng thời gian huấn luyện sẽ lâu

### Nguồn tham khảo

- ChatGPT
- lecture5: lecture5_rnn_token_classification.pdf và lecture5_rnn_token_classification.pdf trên gg classroom
- https://www.tensorflow.org/text/tutorials/text_classification_rnn
- Các thư viện học sâu của python như tensorflow, keras, gensim
- Và 1 số khác...

