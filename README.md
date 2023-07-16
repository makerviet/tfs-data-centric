# Cuộc đua số 2023 - Thử thách: AI tập trung vào dữ liệu

Trong các bài toán thực tế, dữ liệu đóng vai trò vô cùng quan trọng trong việc xây dựng các hệ thống AI có độ chính xác cao và khả năng áp dụng thực tiễn. Nhận thức được điều này, nhiều công ty và tổ chức đã chuyển hướng phát triển AI theo hướng data-centric AI (AI tập trung vào dữ liệu), trong đó dữ liệu được coi là trung tâm và trọng tâm của quá trình xây dựng mô hình (xem thêm tại: [Data Centric AI - Landing AI](https://landing.ai/data-centric-ai/)). Trong thử thách **AI tập trung vào dữ liệu**, các đội chơi sẽ thể hiện khả năng trong việc xây dựng một bộ dữ liệu đủ tốt, đa dạng và đại diện để huấn luyện mô hình AI có độ chính xác cao cho bài toán phát hiện biển báo.

![Data Centric AI vs Model Centric AI - Image from https://towardsdatascience.com/what-are-the-data-centric-ai-concepts-behind-gpt-models-a590071bb727](docs/data-centric-ai.png)

**Data-centric AI vs. model-centric AI - Hình ảnh từ [towardsdatascience](https://towardsdatascience.com/what-are-the-data-centric-ai-concepts-behind-gpt-models-a590071bb727).**

**Mã nguồn và cách đánh giá**

Mã nguồn và các siêu tham số của mô hình sẽ được cung cấp sẵn và cố định. Điều này đảm bảo các đội chơi có cùng một công cụ để huấn luyện và đánh giá mô hình. Các đội chơi sẽ phải tập trung vào việc xây dựng một bộ dữ liệu đủ tốt để huấn luyện mô hình AI có độ chính xác tốt nhất, với phần cứng hạn chế (số lượng iterations được hạn chế). Độ chính xác của mô hình AI sẽ được đánh giá trên tập dữ liệu bí mật (private test) mà các đội chơi khồng được tiếp xúc trước đó.

Data-centric AI đặt dữ liệu làm trọng tâm bởi vì một mô hình AI chỉ có thể hiểu và học từ những gì nó đã thấy. Để có một hệ thống AI chính xác, chúng ta cần xây dựng một tập dữ liệu phát hiện biển báo đa dạng, đại diện cho các tình huống thực tế và có khả năng tổng quát hóa. Các đội chơi sẽ phải đối mặt với các thách thức như thu thập dữ liệu, tiền xử lý và gán nhãn để tạo ra một tập dữ liệu chất lượng cao.

## 1. Dữ liệu mẫu và mã nguồn huấn luyện

- Kiến trúc mô hình: YOLOX Nano
- Kiến trúc mã nguồn
- Dữ liệu mẫu

## 2. Huấn luyện và đánh giá mô hình đã huấn luyện trên môi trường local

- Các bước huấn luyện
- Các bước đánh giá
- NOTE: Không được thay đổi các siêu tham số của mô hình

## 3. Tải lên mô hình và đánh giá trên tập dữ liệu bí mật

- Các bước tải lên
- Các bước đánh giá

## 4. Phương pháp xây dựng bộ dữ liệu

- Các bước thu thập, gán nhãn dữ liệu
- Các bước tiền xử lý dữ liệu
- Gán nhãn dữ liệu với AnyLabeling hoặc labelme. Gợi ý các công cụ khác như CVAT, Label Studio, ...
- Bí quyết để có bộ dữ liệu tốt
