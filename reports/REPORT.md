# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Tuấn Minh

Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Camera trong bộ dữ liệu đứng cố định tại một vị trí và một chiếc xe có thể xuất hiện trong hình trong vài giây liên tiếp. Vì vậy tập chưa gán nhãn và tập kiểm thử cần được chia theo trục thời gian, đồng thời có một khoảng đệm ở giữa để hạn chế các ảnh gần giống nhau xuất hiện ở cả hai tập. Nếu chia ngẫu nhiên, các frame của cùng một chiếc xe hoặc cùng một cảnh có thể vừa được dùng cho AI học, vừa được dùng để kiểm tra. Khi đó mô hình thực chất đã nhìn thấy một cảnh rất giống ảnh kiểm thử trong lúc huấn luyện và điểm đánh giá có thể cao hơn khả năng thực tế của mô hình.

## 2. Mô hình khởi đầu lạnh (cold start)

Ở vòng 0, mô hình yolov8n cold start chưa sử dụng ảnh huấn luyện của bài lab. AP50 đạt 0.771, precision tại ngưỡng confidence 0.25 là 0.925, recall là 0.489 và F1 là 0.640. Recall theo kích thước lần lượt là 0.182 đối với xe nhỏ, 0.547 đối với xe vừa và 0.561 đối với xe lớn. Như vậy mô hình bỏ sót xe nhỏ và xe ở xa nhiều hơn rõ rệt so với các xe có kích thước vừa hoặc lớn.

Quan sát `compare_round0.jpg` cũng cho thấy vấn đề này. Ví dụ ở frame_0050, mô hình có 11 true positive, 2 false positive và 7 false negative. Ở cụm xe nhỏ phía xa gần phần trên của ảnh có các khung dự đoán và khung tham chiếu không khớp hoàn toàn, đồng thời một số xe chỉ hiện thành các điểm sáng nhỏ nên bị bỏ sót. Các frame khác như frame_0150 và frame_0350 cũng còn nhiều false negative.

Tuy nhiên không thể coi toàn bộ false negative hoặc khung lệch là lỗi chắc chắn của mô hình. Nhãn tham chiếu dùng để chấm cũng được tạo bằng máy và chưa được người kiểm tra từng box. Đặc biệt với xe ở rất xa, chỉ nhìn thấy hai đèn hoặc ranh giới thân xe không rõ thì cần người rà lại nhãn tham chiếu trước khi kết luận mô hình dự đoán sai.

## 3. Chiến lược chọn mẫu

Mỗi ảnh trong pool được tính một điểm theo công thức `score = W_U·U + W_A·A + W_D·D`. Trong bài này, một nửa điểm đến từ độ không chắc chắn của mô hình, ba phần mười điểm phản ánh việc ảnh có nhiều box mà mô hình còn lưỡng lự, và hai phần mười còn lại dùng để khuyến khích sự đa dạng theo thời gian. `MIN_GAP_S` được đặt là 2 giây để tránh lấy quá nhiều frame gần như giống hệt nhau, vì camera đứng yên và các frame sát nhau thường chứa cùng một nhóm xe.

Ví dụ `frame_0182.jpg` ở 72.8 giây có điểm 0.9591 và được chọn, `frame_0099.jpg` ở 39.6 giây có điểm 0.9063 và được chọn, còn `frame_0107.jpg` ở 42.8 giây có điểm 0.8876 và cũng được chọn. Ngược lại, `frame_0372.jpg` có điểm khá cao 0.9101 nhưng không được chọn. Frame này ở 148.8 giây, chỉ cách `frame_0369.jpg` ở 147.6 giây khoảng 1.2 giây, trong khi `frame_0369.jpg` đã được chọn. Vì hai ảnh rất gần nhau về thời gian nên sửa cả hai có thể tốn thêm công gán nhãn nhưng không bổ sung nhiều thông tin mới.

Điểm cao chỉ cho biết đây là ảnh mà mô hình đang không chắc hoặc có nhiều dự đoán cần xem lại. Nó không chứng minh rằng sửa ảnh đó rồi fine-tune sẽ chắc chắn làm mô hình tốt hơn. Hiệu quả cuối cùng vẫn phải được kiểm tra trên cùng một tập test độc lập.

## 4. Các vòng học chủ động (active learning)

Kết quả hai vòng như sau:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 345 | 0.504 | -0.267 | 1.000 | 0.104 | 0.189 | 0.000 | 0.108 | 0.244 |

Ở vòng 1, model ban đầu đề xuất 169 box trên 12 ảnh. Sau khi rà lại, bộ nhãn có 345 box. Trong quá trình sửa tôi giữ nguyên 99 box, chỉnh lại 52 box, xóa 18 box false positive và thêm mới 194 box mà pre-label đã bỏ sót. Accept rate chỉ là 59%, cho thấy pre-label ban đầu cần khá nhiều can thiệp của người gán nhãn.

Sau khi fine-tune, AP50 giảm từ 0.771 xuống 0.504, tức giảm khoảng 0.267 so với cold start và cũng giảm 0.267 so với vòng trước vì vòng 0 là vòng trước trực tiếp. Recall giảm mạnh từ 0.489 xuống 0.104. Xe nhỏ giảm từ 0.182 xuống 0.000, xe vừa giảm từ 0.547 xuống 0.108 và xe lớn giảm từ 0.561 xuống 0.244. Precision lại tăng từ 0.925 lên 1.000, nhưng điều này không có nghĩa mô hình tốt hơn vì mô hình đang dự đoán ít xe hơn rất nhiều, dẫn tới ít false positive nhưng số false negative tăng mạnh.

Điều này cũng nhìn thấy rõ trong hai ảnh so sánh. Với frame_0050, cold start đạt TP 11, FP 2 và FN 7, nhưng sau vòng 1 chỉ còn TP 1, FP 0 và FN 17. Nhiều xe mà cold start còn phát hiện được đã biến mất khỏi kết quả vòng 1. Hiện tượng tương tự xuất hiện ở frame_0150, khi TP giảm từ 10 xuống 1 và FN tăng từ 10 lên 19. Vì vậy ở vòng này fine-tune làm mô hình trở nên quá dè dặt và bỏ sót nhiều xe hơn. Một giả thuyết cần kiểm tra là việc fine-tune chỉ với 12 ảnh và 345 box đã làm mô hình thích nghi quá mạnh với lô nhỏ này; ngoài ra cũng cần kiểm tra lại cấu hình train, class mapping và quá trình chuyển nhãn trước khi kết luận nguyên nhân.

Ba nguồn thông tin cần được tách biệt. Trong `BLIND_SCAN.md`, trước khi xem pre-label tôi quan sát độc lập frame_0312 và đếm được 24 xe, đồng thời ghi nhận xe xa nhất ở làn bên phải và xe gần nhất ở làn bên trái là hai vị trí dễ bị AI bỏ sót hoặc vẽ sai. Trong `REVIEW_LOG.csv`, tôi ghi lại các lỗi pre-label cụ thể đã sửa, gồm thêm box cho xe gần mép trái ở frame_0099, chỉnh box quá lớn ở frame_0107 và xóa một box không có đối tượng thật ở frame_0326. Còn `round1_diff.md` mô tả tổng lượng chỉnh sửa nhãn trên cả 12 ảnh. Những dữ liệu này phản ánh quá trình con người rà nhãn, khác với AP50 và recall là kết quả của mô hình sau khi được train lại.

Một ca khó theo guideline là xe ở rất xa, khi trong ảnh gần như chỉ còn hai chấm đèn và rất ít thông tin về thân xe. Trường hợp này dễ bị AI bỏ sót và người gán nhãn cũng khó xác định ranh giới box thống nhất. Bài đánh giá còn bỏ qua các box có chiều cao dưới 16 pixel, vì vậy việc xử lý những xe cực nhỏ cần tuân thủ cùng một quy tắc để tránh tạo nhiễu cho nhãn.

## 5. Kết luận và giới hạn

Sau vòng học chủ động đầu tiên, kết quả chưa tốt hơn cold start. AP50 giảm từ 0.771 xuống 0.504, còn recall giảm từ 0.489 xuống 0.104. Vì vậy tôi chọn dừng tại vòng 1 và chưa train vòng tiếp theo ngay. Trước hết cần kiểm tra lại các box đã sửa, dữ liệu đầu vào và cấu hình fine-tune để tìm nguyên nhân khiến mô hình bỏ sót nhiều xe sau khi học lại.

Hai trường hợp tôi muốn ưu tiên kiểm tra nếu có vòng sau là xe rất xa chỉ còn thấy hai chấm đèn và xe nằm sát hoặc bị cắt bởi mép ảnh. Đây là những trường hợp dễ gây bất định cho cả mô hình và người gán nhãn. Tuy nhiên việc rà thêm ảnh cũng tốn thời gian, vì mỗi frame có thể chứa hàng chục xe. Ngoài ra không nên chọn nhiều frame chỉ cách nhau khoảng một hoặc hai giây nếu chúng gần như là cùng một cảnh, vì phải sửa nhiều box nhưng lượng thông tin mới thu được thấp.

Kết luận của bài này cũng có một số giới hạn. Tập kiểm thử chỉ gồm 20 ảnh với 403 box tham chiếu và có 14 box quá nhỏ dưới 16 pixel bị bỏ qua khi chấm. Nhãn tham chiếu lại do mô hình tạo và chưa được người rà thủ công hoàn toàn. Vì vậy mức AP50 đo được trên tập nhỏ này chưa thể đại diện chắc chắn cho khả năng tổng quát của mô hình trên các video giao thông khác.

Do AP50 của vòng 1 giảm, việc đầu tiên tôi sẽ làm trước khi train thêm là xem lại các box đã chỉnh sửa, đặc biệt các box thêm mới, box ở xe xa và xe sát mép. Sau đó tôi sẽ kiểm tra class mapping, định dạng nhãn, quá trình tạo tập train và cấu hình fine-tune. Chỉ sau khi chắc chắn dữ liệu và pipeline đúng thì mới nên chọn thêm ảnh cho vòng học chủ động tiếp theo.