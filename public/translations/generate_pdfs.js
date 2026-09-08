const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const baseDir = '/home/ubuntu/dr-vantruong-3d-studio/public/translations';
const p1ImgDir = 'file://' + path.join(baseDir, 'images/part1');
const p2ImgDir = 'file://' + path.join(baseDir, 'images/part2');

const css = `
@page {
  size: A4 portrait;
  margin: 14mm 14mm 16mm 14mm;
}

* {
  box-sizing: border-box;
}

body {
  font-family: 'Times New Roman', Times, serif;
  font-size: 9.5pt;
  line-height: 1.35;
  color: #111;
  text-align: justify;
  margin: 0;
  padding: 0;
}

.header-banner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2px solid #004c87;
  padding-bottom: 4px;
  margin-bottom: 12px;
}

.header-tag {
  background-color: #004c87;
  color: white;
  font-size: 10pt;
  font-weight: bold;
  padding: 3px 10px;
  letter-spacing: 0.8px;
}

.journal-ref {
  font-size: 8.5pt;
  color: #555;
  font-style: italic;
}

h1.article-title {
  font-size: 16pt;
  font-weight: bold;
  line-height: 1.25;
  margin-top: 4px;
  margin-bottom: 10px;
  color: #0c2340;
}

.author-block {
  font-size: 9pt;
  margin-bottom: 12px;
  line-height: 1.3;
}

.author-names {
  font-weight: bold;
  color: #222;
}

.author-affiliation {
  font-style: italic;
  color: #555;
  font-size: 8.5pt;
}

.abstract-box {
  background-color: #f3f7fb;
  border-left: 3.5px solid #004c87;
  padding: 9px 12px;
  margin-bottom: 14px;
  font-size: 8.8pt;
  line-height: 1.32;
}

.abstract-box strong {
  color: #004c87;
}

.two-col {
  column-count: 2;
  column-gap: 18px;
  column-rule: 1px solid #e2e8f0;
}

.first-letter {
  float: left;
  font-size: 34pt;
  line-height: 0.8;
  font-weight: bold;
  color: #004c87;
  padding-right: 4px;
  padding-top: 2px;
  font-family: 'Times New Roman', serif;
}

h2.section-heading {
  font-size: 10.5pt;
  font-weight: bold;
  color: #004c87;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: 12px;
  margin-bottom: 6px;
  border-bottom: 1px solid #004c87;
  padding-bottom: 2px;
  break-after: avoid;
}

p {
  margin-top: 0;
  margin-bottom: 6px;
  text-indent: 14px;
}

p.no-indent {
  text-indent: 0;
}

.figure-box {
  break-inside: avoid;
  margin: 10px 0;
  text-align: center;
}

.figure-box img {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 2px;
}

.figure-caption {
  font-size: 8pt;
  line-height: 1.25;
  color: #222;
  margin-top: 4px;
  text-align: justify;
}

.figure-caption strong {
  color: #004c87;
}

.table-box {
  break-inside: avoid;
  margin: 10px 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 8pt;
  line-height: 1.22;
  margin-top: 4px;
  margin-bottom: 4px;
}

th {
  background-color: #004c87;
  color: white;
  font-weight: bold;
  padding: 4px 5px;
  text-align: left;
  border: 1px solid #004c87;
}

td {
  padding: 3px 5px;
  border: 1px solid #ccc;
  vertical-align: middle;
}

tr:nth-child(even) td {
  background-color: #f9fbfd;
}

.table-title {
  font-size: 8.2pt;
  font-weight: bold;
  color: #004c87;
  margin-bottom: 2px;
}

.table-footnote {
  font-size: 7.2pt;
  color: #666;
  font-style: italic;
  margin-top: 2px;
}

.citation-block {
  font-size: 8pt;
  line-height: 1.25;
  color: #444;
  margin-top: 12px;
  padding-top: 6px;
  border-top: 1px solid #ccc;
  break-inside: avoid;
}
`;

// ==================== PART 1 HTML ====================
const part1Html = `<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Giải phẫu Cơ cau mày: Phần I. Địa hình học Cơ cau mày</title>
<style>${css}</style>
</head>
<body>

<div class="header-banner">
  <span class="header-tag">PHẪU THUẬT TẠO HÌNH THẨM MỸ</span>
  <span class="journal-ref">Plast. Reconstr. Surg. 120: 1647, 2007 • Bản dịch tiếng Việt chuyên khoa Y</span>
</div>

<h1 class="article-title">Giải phẫu Cơ cau mày:<br>Phần I. Địa hình học Cơ cau mày</h1>

<div class="author-block">
  <div class="author-names">Jeffrey E. Janis, M.D.; Ashkan Ghavami, M.D.; Joshua A. Lemmon, M.D.; Jason E. Leedy, M.D.; Bahman Guyuron, M.D.</div>
  <div class="author-affiliation">Khoa Phẫu thuật Tạo hình, Trung tâm Y tế Southwestern - Đại học Texas, Dallas, Texas; và Khoa Phẫu thuật Tạo hình & Tái tạo, Trường Y Đại học Case Western Reserve, Cleveland, Ohio.</div>
</div>

<div class="abstract-box">
  <p class="no-indent"><strong>TỔNG QUAN:</strong> Phẫu thuật cắt bỏ hoàn toàn cơ cau mày (corrugator supercilii muscle - CSM) đóng vai trò then chốt trong điều trị ngoại khoa chứng đau nửa đầu (migraine) và giúp phòng ngừa các biến dạng bất thường sau phẫu thuật trẻ hóa vùng trán. Các phân tích chi tiết về địa hình kích thước cơ cau mày cũng như mối liên hệ chặt chẽ của nó với mạng lưới phân nhánh dây thần kinh trên ổ mắt trước đây chưa được mô tả đầy đủ. Phần I của nghiên cứu này nhằm xác định bản đồ địa hình cơ cau mày dựa trên các mốc xương cố định bên ngoài.</p>
  <p class="no-indent" style="margin-top:4px;"><strong>PHƯƠNG PHÁP:</strong> Phẫu tích 25 đầu xác tươi (gồm 50 cơ cau mày và 50 dây thần kinh trên ổ mắt) nhằm bộc lộ và tách biệt hoàn toàn cơ cau mày khỏi các cơ lân cận. Tiến hành các phép đo chuẩn hóa về kích thước cơ cau mày dựa theo mốc điểm nasion (gốc mũi) và bờ ngoài ổ mắt (lateral orbital rim - LOR).</p>
  <p class="no-indent" style="margin-top:4px;"><strong>KẾT QUẢ:</strong> So với điểm nasion, nguyên ủy trong cùng của cơ cau mày nằm ở vị trí 2,9 ± 1,0 mm; nguyên ủy ngoài cùng nằm ở 14,0 ± 2,8 mm. Điểm bám tận ngoài cùng của cơ cau mày đo được 43,3 ± 2,9 mm tính từ nasion (tương ứng 85% khoảng cách từ nasion đến bờ ngoài ổ mắt) hoặc cách bờ ngoài ổ mắt 7,6 ± 2,7 mm về phía trong. Điểm cực trên (đỉnh cơ - apex) nằm cách mặt phẳng nasion - bờ ngoài ổ mắt 32,6 ± 3,1 mm về phía đầu (phía trên) và cách bờ ngoài ổ mắt 18,0 ± 3,7 mm về phía trong. Không có sự khác biệt có ý nghĩa thống kê giữa hai bên mắt trái và phải.</p>
  <p class="no-indent" style="margin-top:4px;"><strong>KẾT LUẬN:</strong> Kích thước thực tế của cơ cau mày trải rộng hơn nhiều so với các y văn mô tả trước đây và có thể định vị dễ dàng dựa vào các mốc xương cố định. Dữ liệu này giúp phẫu thuật viên thực hiện cắt bỏ cơ cau mày an toàn, triệt để và đối xứng trong phẫu thuật trẻ hóa trán, đồng thời giải ép hiệu quả nhánh thần kinh trên ổ mắt và trên ròng rọc trong điều trị đau nửa đầu. (<em>Plast. Reconstr. Surg. 120: 1647, 2007.</em>)</p>
</div>

<div class="two-col">

<p class="no-indent"><span class="first-letter">P</span>hẫu thuật cắt bỏ toàn bộ cơ cau mày đã được chỉ định rộng rãi cho cả mục đích trẻ hóa vùng trán lẫn điều trị ngoại khoa chứng đau nửa đầu (migraine).<sup>1–3</sup> Việc cắt bỏ cơ cau mày không đồng đều hoặc cắt bỏ không hoàn toàn sau phẫu thuật trẻ hóa trán có thể dẫn đến các di chứng không mong muốn như lồi lõm, tạo hố lõm cục bộ và còn sót hoạt tính co cơ cau mày, khiến các nếp nhăn động vùng gian mày tiếp tục tồn tại dai dẳng.<sup>1–3</sup> Tất cả những biến chứng này có thể lộ rõ hơn khi bệnh nhân biểu cảm hoặc cử động cơ vùng trán.<sup>1,4,5</sup> Nguyên nhân có thể bắt nguồn từ đường tiếp cận phẫu thuật, kỹ thuật thao tác cụ thể hoặc kinh nghiệm của phẫu thuật viên đối với từng phương pháp.<sup>1,6,7</sup></p>

<p>Trong một nghiên cứu gần đây, Walden và cộng sự<sup>6</sup> đã chỉ ra rằng mức độ cắt bỏ cơ cau mày có sự dao động lớn tùy thuộc vào đường mổ, trong đó có tới 1/3 thân ngang của cơ cau mày vẫn còn sót lại sau những nỗ lực cắt cơ qua đường rạch mí mắt (transpalpebral approach). Mặc dù tác giả chính (B.G.) cho rằng điều này chủ yếu phụ thuộc vào kỹ thuật,<sup>7,8</sup> việc nắm vững kích thước giải phẫu bình thường của cơ cau mày dựa trên các mốc xương cố định sẽ giúp giảm thiểu tính bất định này và mở đường cho một quy trình cắt cơ (myectomy) chuẩn xác, có hệ thống. Ngoài ra, hiểu biết toàn diện về kích thước cơ cau mày còn hỗ trợ đắc lực cho các phẫu thuật viên ít kinh nghiệm đạt được kết quả thành công mỹ mãn khi thực hiện bất kỳ đường mổ trẻ hóa trán nào trong số rất nhiều phương pháp đã được công bố.</p>

<p>Đau nửa đầu từ lâu đã được giả thuyết là có liên quan đến các điểm kích hoạt thần kinh ngoại biên (trigger points).<sup>9–14</sup> Các dây thần kinh trên ổ mắt (supraorbital) và trên ròng rọc (supratrochlear) đã được xác định là một trong bốn vị trí kích hoạt ngoại biên chịu trách nhiệm cho các triệu chứng đau nửa đầu.<sup>10</sup> Sự thuyên giảm hoặc khỏi hoàn toàn cơn đau nửa đầu đã được chứng minh sau khi bất hoạt cơ cau mày bằng độc tố Botulinum type A (Botox),<sup>9,10</sup> với cơ chế được lý giải là nhờ giải ép dây thần kinh trên ổ mắt và trên ròng rọc khi các cơ bao quanh được thư giãn.<sup>10,14</sup> Thành công lâu dài đã được ghi nhận trên đại đa số bệnh nhân sau đó được tiến hành phẫu thuật cắt bỏ toàn bộ cơ cau mày.<sup>10</sup></p>

<p>Dựa trên quan sát sâu rộng trong phẫu thuật, tác giả chính (B.G.) nhận định rằng các nhánh thần kinh trên ổ mắt đan xen và phân bố phức tạp hơn nhiều bên trong các sợi cơ cau mày, củng cố sự cần thiết phải cắt bỏ an toàn và triệt để cơ cau mày nhằm giải ép thần kinh trên ổ mắt trong điều trị đau nửa đầu. Mặc dù địa hình bao bọc của các điểm kích hoạt ngoại biên khác đã được mô tả,<sup>9,11–13</sup> mối quan hệ giải phẫu mật thiết giữa dây thần kinh trên ổ mắt và các sợi cơ cau mày vẫn đòi hỏi phải được khảo sát kỹ lưỡng hơn. Hiểu biết tường tận về kích thước cơ cau mày và tương quan của nó với mạng lưới phân nhánh thần kinh trên ổ mắt sẽ nâng cao độ an toàn và tính chuẩn xác của phẫu thuật trẻ hóa trán và điều trị đau nửa đầu. Trong Phần I của nghiên cứu này, các kích thước địa hình của cơ cau mày dựa trên các mốc xương cố định bên ngoài sẽ được xác lập; trong khi Phần II sẽ mô tả chi tiết các dạng phân nhánh của dây thần kinh trên ổ mắt liên quan đến các sợi cơ cau mày.</p>

<h2 class="section-heading">GIẢI PHẪU LIÊN QUAN</h2>

<p class="no-indent">Cơ cau mày là một trong ba nhóm cơ hạ cung mày thường được mô tả (bao gồm phần trong cơ vòng mi và cơ hạ mày) và được cấu tạo gồm hai đầu (hai thân cơ). Thân ngang (transverse head) bắt nguồn từ bờ trên-trong ổ mắt và chạy ngang ra ngoài để bám tận vào lớp hạ bì ở 1/3 giữa cung mày, đồng thời đan xen chằng chịt với các sợi cơ vòng mi (orbicularis oculi) và cơ trán (frontalis). Thân chéo (oblique head) của cơ cau mày có kích thước nhỏ hơn, với các sợi cơ thường chạy song song với sợi cơ hạ mày (depressor supercilii) sau khi bám tận vào phần trong cung mày.</p>

<p>Knize<sup>15</sup> đã phẫu tích 40 nửa đầu xác để đánh giá giải phẫu cơ chi tiết vùng trán. Nguyên ủy của cơ cau mày được nhận thấy là hằng định và nằm tại xương trán gần bờ trên-trong ổ mắt, ở phía trước và hơi chếch lên trên so với ròng rọc của cơ chéo trên nhãn cầu.<sup>15</sup> Các sợi cơ cau mày sau đó chạy chếch lên trên và ra ngoài (superolaterally), xuyên qua cơ trán và cơ vòng mi trước khi bám tận vào da nửa trong cung mày.<sup>15</sup> Cơ cau mày cũng đi xuyên qua lớp đệm mỡ dưới cân galea (galeal fat pad) trước khi chia các nhánh bám vào lớp trung bì da.<sup>15</sup> Tuy nhiên, từ các mô tả này vẫn chưa rõ liệu cơ cau mày có mở rộng vượt ra ngoài đường dính thái dương (temporal fusion line) hay không và vị trí bám tận ngoài cùng chính xác nằm ở đâu.<sup>16,17</sup></p>

<p>Trong một báo cáo gần đây so sánh hiệu quả của đường mổ qua mí mắt, phẫu thuật nội soi và đường mổ mở coronal, thân ngang của cơ cau mày thường bị cắt bỏ không hoàn toàn, chủ yếu ở phần ngoài khi mổ qua đường mí mắt.<sup>6</sup> Các tác giả nhận thấy thân ngang của cơ cau mày dài và dày hơn (trung bình 7,5 mm) so với thân chéo (2,0 mm).<sup>6</sup> Khả năng cắt bỏ cơ cau mày triệt để hơn cùng với cơ hạ mày và cơ tháp (procerus) đạt được rõ rệt nhất qua phẫu thuật nội soi. Ngoài ra, màu sắc cơ là một dấu hiệu thị giác hữu ích để phân biệt các bó cơ: ví dụ phần trong cơ vòng mi nằm nông và có màu hồng nhạt hơn so với các sợi cơ hạ mày màu đỏ chạy dọc.<sup>6</sup> Isse và Elahi<sup>17</sup> thực hiện nghiên cứu trên số lượng xác nhỏ hơn và thấy rằng ở 2/3 ngoài cung mày, sợi cơ cau mày xuyên qua cơ vòng mi và cơ trán. Vùng ngoài cùng này tuy khó phẫu tích nhưng rất quan trọng vì nó có thể là nguyên nhân chính gây ra vết lõm cung mày phía ngoài sau mổ.<sup>17,18</sup></p>

<p>Nhánh vận động cho cơ cau mày xuất phát từ nhánh trán của dây thần kinh mặt (dây VII), trong khi nhánh gò má có vẻ phân nhánh vận động cho thân chéo.<sup>2</sup> Việc quan sát thấy cơ cau mày tái hoạt động trở lại sau khi cắt phần ngoài cung cấp bằng chứng cho thấy có tồn tại các sợi thần kinh vận động xuất phát từ phía trong tham gia vào quá trình tái chi phối thần kinh.<sup>1,2,15</sup> Điều này càng khẳng định tầm quan trọng của việc cắt bỏ hoàn toàn cơ cau mày để giải ép thần kinh trên ổ mắt/trên ròng rọc.</p>

<p>Mối liên hệ giữa dây thần kinh trên ròng rọc và cơ cau mày đã được biết rõ: thần kinh thoát ra ngay phía ngoài nguyên ủy cơ cau mày, đi vào cơ (chia thành 3–4 nhánh nhỏ), chạy hướng lên trên ngay dưới bề mặt trước của cơ cau mày rồi xuyên qua cơ trán.<sup>2,19</sup> Tuy nhiên, mối liên hệ mật thiết giữa dây thần kinh trên ổ mắt với cơ cau mày trước đây chưa được làm sáng tỏ.</p>

<div class="figure-box">
  <img src="${p1ImgDir}/fig1.jpg" alt="Hình 1">
  <div class="figure-caption"><strong>Hình 1.</strong> Bản đồ địa hình cơ cau mày (CSM); hình ảnh phẫu tích trên xác với các khoảng cách trung bình. Phẫu tích cơ cau mày thể hiện các mốc dữ liệu chính định vị kích thước cơ tương quan với điểm nasion (N) và bờ ngoài ổ mắt (LOR).</div>
</div>

<h2 class="section-heading">ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU</h2>

<p class="no-indent">Hai mươi lăm đầu xác tươi (gồm 50 cơ cau mày và 50 dây thần kinh trên ổ mắt) được phẫu tích bằng đường rạch hình chữ thập tập trung tại gốc mũi (radix), với nhánh ngang đi dọc theo vòm cung mày. Cơ trán và cơ hạ mày được bóc tách tỉ mỉ khỏi cơ cau mày và nâng lên cùng với vạt da. Khi toàn bộ phạm vi thân ngang và thân chéo của cơ cau mày đã lộ rõ, điểm nasion và đỉnh bờ ngoài ổ mắt (điểm xương nhô ra ngoài cùng) được đánh dấu và tiến hành các phép đo chuẩn hóa.</p>

<p>Do hiện tượng biến dạng nhãn cầu và thay đổi mô mềm ở tử thi, các mốc mô mềm không được sử dụng. Thay vào đó, các mốc xương cố định được chọn làm điểm quy chiếu chuẩn. Kích thước cơ theo chiều đứng được đo tương quan với đường ngang nối điểm nasion và bờ ngoài ổ mắt. Kích thước cơ theo chiều ngang được đo tương quan với đường thẳng đứng đi qua nasion, gai mũi trước và điểm menton (cằm). Các giá trị được trình bày dưới dạng giá trị trung bình kèm độ lệch chuẩn (Mean ± SD) và so sánh giữa hai bên bằng kiểm định paired t-test.</p>

<div class="figure-box">
  <img src="${p1ImgDir}/fig2.jpg" alt="Hình 2">
  <div class="figure-caption"><strong>Hình 2.</strong> Tổng hợp kích thước toàn diện của cơ cau mày. Bản vẽ minh họa theo tỷ lệ chuẩn tất cả các điểm đo đạc của cơ cau mày tương quan với các mốc xương có thể sờ thấy [nasion (N) và bờ ngoài ổ mắt (LOR)]. Lưu ý việc tách bóc các sợi cơ đan xen là cần thiết để bộc lộ hết giới hạn ngoài cùng của cơ.</div>
</div>

<div class="table-box">
  <div class="table-title">Bảng 1. Kích thước địa hình cơ cau mày theo các mốc quy chiếu điểm Nasion (Giá trị trung bình)</div>
  <table>
    <thead>
      <tr>
        <th>Mốc đo</th>
        <th>Phải (mm)</th>
        <th>Trái (mm)</th>
        <th>Tổng thể (mm)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Nasion đến Bờ ngoài ổ mắt</td>
        <td>50,9 ± 2,9</td>
        <td>50,8 ± 2,8</td>
        <td>50,8 ± 2,9</td>
      </tr>
      <tr>
        <td>Nasion đến Nguyên ủy trong</td>
        <td>2,7 ± 0,9</td>
        <td>3,0 ± 1,0</td>
        <td>2,9 ± 1,0</td>
      </tr>
      <tr>
        <td>Nasion đến Nguyên ủy ngoài</td>
        <td>13,8 ± 3,0</td>
        <td>14,2 ± 2,6</td>
        <td>14,0 ± 2,8</td>
      </tr>
      <tr>
        <td>Nasion đến Giới hạn ngoài cùng</td>
        <td>44,0 ± 2,6</td>
        <td>42,7 ± 3,5</td>
        <td>43,3 ± 2,9</td>
      </tr>
      <tr>
        <td>Nasion đến Đỉnh cơ (Apex)*</td>
        <td>33,0 ± 2,8</td>
        <td>32,5 ± 2,0</td>
        <td>32,9 ± 2,6</td>
      </tr>
    </tbody>
  </table>
  <div class="table-footnote">*Điểm mở rộng cao nhất về phía đầu của các sợi cơ cau mày.</div>
</div>

<div class="table-box">
  <div class="table-title">Bảng 2. Kích thước địa hình cơ cau mày theo các mốc quy chiếu Bờ ngoài ổ mắt (LOR) (Giá trị trung bình)</div>
  <table>
    <thead>
      <tr>
        <th>Mốc đo</th>
        <th>Phải (mm)</th>
        <th>Trái (mm)</th>
        <th>Tổng thể (mm)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>LOR đến Đỉnh cơ (Apex)</td>
        <td>17,9 ± 3,3</td>
        <td>18,0 ± 2,5</td>
        <td>18,0 ± 3,7</td>
      </tr>
      <tr>
        <td>LOR đến Giới hạn ngoài cùng</td>
        <td>6,9 ± 2,6</td>
        <td>8,3 ± 3,2</td>
        <td>7,6 ± 2,7</td>
      </tr>
    </tbody>
  </table>
  <div class="table-footnote">LOR: Điểm trong cùng sờ thấy được của bờ ngoài ổ mắt.</div>
</div>

<div class="table-box">
  <div class="table-title">Bảng 3. Kích thước địa hình cơ cau mày theo phương thẳng đứng (Giá trị trung bình)</div>
  <table>
    <thead>
      <tr>
        <th>Mặt phẳng quy chiếu* đến:</th>
        <th>Phải (mm)</th>
        <th>Trái (mm)</th>
        <th>Tổng thể (mm)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Nguyên ủy Trong - Dưới</td>
        <td>9,8 ± 2,1</td>
        <td>9,3 ± 2,0</td>
        <td>9,8 ± 2,2</td>
      </tr>
      <tr>
        <td>Nguyên ủy Trong - Trên</td>
        <td>18,6 ± 2,4</td>
        <td>18,2 ± 2,5</td>
        <td>18,7 ± 2,4</td>
      </tr>
      <tr>
        <td>Giới hạn Ngoài - Dưới</td>
        <td>22,3 ± 3,2</td>
        <td>21,1 ± 3,7</td>
        <td>21,9 ± 3,3</td>
      </tr>
      <tr>
        <td>Giới hạn Ngoài - Trên</td>
        <td>28,9 ± 2,8</td>
        <td>26,8 ± 3,5</td>
        <td>28,8 ± 3,5</td>
      </tr>
      <tr>
        <td>Đỉnh cơ (Apex)</td>
        <td>32,8 ± 3,1</td>
        <td>31,6 ± 2,5</td>
        <td>32,6 ± 3,1</td>
      </tr>
    </tbody>
  </table>
  <div class="table-footnote">*Mặt phẳng đại diện cho đường thẳng nằm ngang đi qua các điểm nasion và bờ ngoài ổ mắt.</div>
</div>

<h2 class="section-heading">KẾT QUẢ</h2>

<p class="no-indent">Không ghi nhận sự khác biệt có ý nghĩa thống kê giữa kích thước cơ cau mày bên phải và bên trái theo phân tích paired t-test (p < 0,0001); do đó số liệu đo đạc bên phải (n = 25) và bên trái (n = 25) được gộp chung (tổng n = 50) để tăng độ mạnh thống kê (Bảng 1, 2, 3 và Hình 1, 2).</p>

<p>Khoảng cách từ nasion đến bờ ngoài ổ mắt đo được 50,8 ± 2,9 mm (dao động 46 đến 59 mm). Điểm bám tận ngoài cùng của cơ cau mày nằm cách nasion 43,3 ± 2,9 mm (chiếm 85% tổng khoảng cách nasion - LOR), tương đương cách bờ ngoài ổ mắt 7,6 ± 2,7 mm về phía trong.</p>

<p>Nguyên ủy phía trong của cơ nằm cách nasion 2,9 ± 1,0 mm, trong khi nguyên ủy phía ngoài cách nasion 14,0 ± 2,8 mm. Chiều rộng trung bình của diện bám nguyên ủy là 11,1 mm. Đỉnh cơ (apex) nằm cách nasion 32,9 ± 2,6 mm theo chiều ngang (hoặc cách bờ ngoài ổ mắt 18,0 ± 3,7 mm về phía trong) và nằm cách mặt phẳng ngang quy chiếu 32,6 ± 3,1 mm theo chiều đứng.</p>

<h2 class="section-heading">BÀN LUẬN</h2>

<p class="no-indent">Cắt bỏ toàn bộ cơ cau mày là yêu cầu bắt buộc cho cả phẫu thuật trẻ hóa trán và điều trị đau nửa đầu.<sup>1–3,6,9</sup> Việc cắt cơ không triệt để hoặc không đều bằng các thủ thuật nạo/cắt giảm thể tích đơn giản có thể dẫn đến hậu quả nghiêm trọng: bất đối xứng trán-mày, lõm da khi cử động và tái phát các nếp nhăn vùng gian mày.<sup>1,2,15</sup> Ngoài ra, hiện tượng tái chi phối thần kinh bất thường của phần cơ còn sót lại có thể tạo nên các "sừng ngoài" (lateral horns) gồ lên chỉ sau 3–4 tháng sau mổ.<sup>4</sup></p>

<div class="figure-box">
  <img src="${p1ImgDir}/fig3.jpg" alt="Hình 3">
  <div class="figure-caption"><strong>Hình 3.</strong> Các kích thước có giá trị ứng dụng lâm sàng cao nhất. Ảnh chụp bệnh nhân minh họa các thông số địa hình thực tế: Dữ liệu theo chiều ngang gồm nguyên ủy trong cùng (2,9 mm); điểm vươn ra ngoài cùng (43,3 mm từ nasion và 7,6 mm từ bờ ngoài ổ mắt); vị trí đỉnh cơ cách bờ ngoài ổ mắt (18 mm). Dữ liệu theo chiều đứng (tính từ mặt phẳng nasion - LOR): đỉnh cơ (32,6 mm) và giới hạn ngoài cùng của cơ cau mày (28,8 mm).</div>
</div>

<p>Trong nghiên cứu chi tiết của Walden và cộng sự,<sup>6</sup> có tới 1/3 thân ngang cơ cau mày bị bỏ sót sau phẫu thuật qua đường mí mắt. Guyuron<sup>7</sup> đã đưa ra 3 nguyên nhân chính: (1) các sợi đan xen với cơ trán và cơ vòng mi cản trở tầm nhìn bờ trên thân ngang; (2) lo ngại làm tổn thương dây thần kinh trên ổ mắt; và (3) đánh giá thấp độ vươn rộng ra phía ngoài của cơ cau mày.</p>

<p>Nghiên cứu của chúng tôi xác định điểm bám tận ngoài cùng của cơ cách nasion tới 43,3 mm (85% khoảng cách tới bờ ngoài ổ mắt). Vùng này mỏng và nằm nông hơn, đan xen phức tạp vào cơ vòng mi và cơ trán, đòi hỏi phẫu tích tỉ mỉ dưới kính lúp độ phóng đại 3.8× để tránh làm tổn thương cơ trán và cơ vòng mi. Chúng tôi cũng không thấy ranh giới phân tách rõ giữa thân ngang và thân chéo, cho thấy cơ cau mày thực chất là một khối cơ thống nhất với các nguyên ủy phụ thay vì hai đầu cơ tách rời.</p>

<h2 class="section-heading">KẾT LUẬN</h2>

<p class="no-indent">Kích thước của cơ cau mày trải rộng ra phía ngoài và lên phía trên nhiều hơn so với các mô tả trước đây, với diện bám nguyên ủy rộng trung bình 11,1 mm. Việc phẫu tích và cắt bỏ hoàn toàn cơ cau mày có thể được thực hiện an toàn, chính xác và có hệ thống nhờ vào các mốc xương cố định bên ngoài, giúp giảm thiểu tối đa các biến chứng thẩm mỹ sau mổ và giải ép thần kinh hiệu quả trong điều trị ngoại khoa chứng đau nửa đầu.</p>

<div class="citation-block">
  <strong>Liên hệ tác giả:</strong> Jeffrey E. Janis, M.D., Department of Plastic Surgery, The University of Texas Southwestern Medical Center, 1801 Inwood Road, WA4.240, Dallas, Texas 75390-9132.<br>
  <strong>Email:</strong> jeffrey.janis@utsouthwestern.edu<br>
  <strong>Lời cảm ơn:</strong> Các tác giả xin chân thành cảm ơn TS.BS. Rod J. Rohrich đã tài trợ các mẫu xác tươi phục vụ nghiên cứu này.
</div>

</div>

</body>
</html>
`;

// ==================== PART 2 HTML ====================
const part2Html = `<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Giải phẫu Cơ cau mày: Phần II. Các dạng phân nhánh Thần kinh trên ổ mắt</title>
<style>${css}</style>
</head>
<body>

<div class="header-banner">
  <span class="header-tag">PHẪU THUẬT TẠO HÌNH THẨM MỸ</span>
  <span class="journal-ref">Plast. Reconstr. Surg. 121: 233, 2008 • Bản dịch tiếng Việt chuyên khoa Y</span>
</div>

<h1 class="article-title">Giải phẫu Cơ cau mày:<br>Phần II. Các dạng phân nhánh Thần kinh trên ổ mắt</h1>

<div class="author-block">
  <div class="author-names">Jeffrey E. Janis, M.D.; Ashkan Ghavami, M.D.; Joshua A. Lemmon, M.D.; Jason E. Leedy, M.D.; Bahman Guyuron, M.D.</div>
  <div class="author-affiliation">Khoa Phẫu thuật Tạo hình, Trung tâm Y tế Southwestern - Đại học Texas, Dallas, Texas; và Phân khoa Phẫu thuật Tạo hình & Tái tạo, Trường Y Đại học Case Western Reserve, Cleveland, Ohio.</div>
</div>

<div class="abstract-box">
  <p class="no-indent"><strong>TỔNG QUAN:</strong> Bài báo này tập trung vào việc mô tả chi tiết các dạng phân nhánh của dây thần kinh trên ổ mắt (supraorbital nerve - SON) trong mối tương quan với các sợi cơ cau mày (corrugator supercilii muscle - CSM) và thiết lập hệ thống 4 kiểu phân nhánh giúp nâng cao hiểu biết giải phẫu học ngoại khoa.</p>
  <p class="no-indent" style="margin-top:4px;"><strong>PHƯƠNG PHÁP:</strong> Hai mươi lăm đầu xác tươi (50 cơ cau mày và 50 dây thần kinh trên ổ mắt) được phẫu tích và bộc lộ cơ cau mày. Sau khi ghi nhận các điểm đo đạc địa hình cơ (ở Phần I), các nhánh thần kinh trên ổ mắt được lần theo từ điểm thoát ra khỏi ổ mắt và bóc tách dọc theo các ranh giới giải phẫu của cơ. Phân tích các dạng phân nhánh liên quan đến sợi cơ và xây dựng bảng phân loại chi tiết.</p>
  <p class="no-indent" style="margin-top:4px;"><strong>KẾT QUẢ:</strong> Xác định được 4 dạng phân nhánh của dây thần kinh trên ổ mắt: Ở <strong>Type I (40%)</strong>, chỉ có nhánh sâu của dây thần kinh trên ổ mắt (SON-D) cho các nhánh chạy trực tiếp dọc theo mặt dưới của cơ cau mày. Ở <strong>Type II (34%)</strong>, xuất hiện các nhánh bắt nguồn trực tiếp từ nhánh nông (SON-S) bên cạnh các nhánh từ nhánh sâu. Ở <strong>Type III (4%)</strong>, chỉ ghi nhận các nhánh riêng biệt từ nhánh nông (SON-S) mà không có nhánh nào từ nhánh sâu. Ở <strong>Type IV (22%)</strong>, sự phân nhánh đáng kể chỉ bắt đầu ở vị trí cao hơn về phía đầu so với khối cơ, do đó không có mối liên hệ trực tiếp với các sợi cơ cau mày.</p>
  <p class="no-indent" style="margin-top:4px;"><strong>KẾT LUẬN:</strong> Trái ngược với các báo cáo trước đây, cả nhánh sâu và nhánh nông của dây thần kinh trên ổ mắt đều có mối quan hệ mật thiết với các sợi cơ cau mày. Đã xác định 4 kiểu phân nhánh và nhận diện các vị trí chèn ép thần kinh tiềm tàng. Thông tin giải phẫu chi tiết này giúp tăng cường tính an toàn và độ chính xác khi thực hiện cắt bỏ toàn bộ cơ cau mày. (<em>Plast. Reconstr. Surg. 121: 233, 2008.</em>)</p>
</div>

<div class="two-col">

<p class="no-indent"><span class="first-letter">T</span>rong phần I của nghiên cứu này, chúng tôi đã công bố dữ liệu về kích thước địa hình của cơ cau mày tương quan với các mốc xương cố định.<sup>1,2</sup> Điều này cho phép tiếp cận phẫu thuật có hệ thống và chuẩn xác hơn khi cắt bỏ cơ cau mày cho cả mục đích trẻ hóa trán và điều trị ngoại khoa đau nửa đầu.<sup>1–7</sup></p>

<p>Đa số các chứng đau nửa đầu được lý giải là có liên quan đến sự kích thích, kẹt hoặc chèn ép tại các điểm kích hoạt thần kinh ngoại biên.<sup>8–13</sup> Dây thần kinh trên ổ mắt và trên ròng rọc được xác định là các vị trí kích hoạt vùng trán.<sup>8,9</sup> Giả thuyết này được củng cố bởi thực tế là triệu chứng đau nửa đầu thuyên giảm rõ rệt ở phần lớn bệnh nhân sau khi làm liệt cơ cau mày bằng Botulinum toxin type A.<sup>8–12,14–16</sup> Mối quan hệ mật thiết giữa dây thần kinh trên ổ mắt và các sợi cơ cau mày đòi hỏi phải được khảo sát sâu hơn.</p>

<h2 class="section-heading">GIẢI PHẪU LIÊN QUAN</h2>

<p class="no-indent">Dây thần kinh trên ổ mắt là một dây thần kinh cảm giác thuần túy xuất phát từ nhánh mắt (V<sub>1</sub>) của dây thần kinh sinh ba (dây V). Thần kinh thoát ra qua khuyết trên ổ mắt (supraorbital notch) trong 90% trường hợp, nhưng cũng có thể thoát ra qua một lỗ xương thực sự (true bony foramen) nằm cách bờ trên ổ mắt 1,5 cm trong 10% trường hợp.<sup>17–20</sup> Beer và cộng sự<sup>21</sup> nghiên cứu và phát hiện 1 điểm thoát duy nhất ở 84% ổ mắt phải và 82% ổ mắt trái; tuy nhiên có nhiều hơn 1 điểm thoát ở 14% bên phải và 16% bên trái.<sup>21</sup></p>

<div class="figure-box">
  <img src="${p2ImgDir}/fig1.jpg" alt="Hình 1">
  <div class="figure-caption"><strong>Hình 1.</strong> Đường rạch hình chữ thập. Các đường vẽ đánh dấu vị trí đường mổ và kế hoạch bóc tách vạt da.</div>
</div>

<p>Sau khi thoát khỏi bờ trên ổ mắt, thân dây thần kinh trên ổ mắt chia thành một nhánh nông (superficial branch - SON-S) và một nhánh sâu (deep branch - SON-D).<sup>17,18,20</sup> Nhánh sâu tách khỏi thân thần kinh tại khuyết trên ổ mắt ở 5% đến 10% bệnh nhân, nhưng cũng có thể thoát ra từ một lỗ riêng biệt nằm chếch ra ngoài hơn.<sup>17</sup> Nhánh sâu có đường đi hằng định hơn, chạy sâu dưới cơ trán hướng lên trên giữa cân galea và màng xương ở nửa dưới trán để chi phối cảm giác cho da đầu vùng trán - đỉnh.<sup>17,18</sup> Đường đi của nó chạy song song và ngay phía trong đường dính thái dương.<sup>18</sup></p>

<p>Nhánh nông chia thành nhiều nhánh nhỏ xuyên qua cơ trán và chạy nông trên bề mặt cơ về phía đường chân tóc để chi phối cảm giác cho trán và da đầu trước (gặp ở 90% mẫu xác của Knize).<sup>17,18</sup></p>

<h2 class="section-heading">ĐỐI TƯỢNG VÀ PHƯƠNG PHÁP NGHIÊN CỨU</h2>

<p class="no-indent">Hai mươi lăm đầu xác tươi (50 cơ cau mày và 50 dây thần kinh trên ổ mắt) được phẫu tích bằng đường rạch chữ thập ở gốc mũi (Hình 1). Nâng vạt mỡ-da dưới độ phóng đại kính lúp 3.8×. Cơ trán và cơ hạ mày được tách sắc bén khỏi cơ cau mày. Cơ cau mày được phẫu tích tỉ mỉ bộc lộ toàn bộ (Hình 2).</p>

<div class="figure-box">
  <img src="${p2ImgDir}/fig2.jpg" alt="Hình 2">
  <div class="figure-caption"><strong>Hình 2.</strong> Phẫu tích trên xác thể hiện phạm vi bộc lộ và phẫu tích khung xương hóa cơ cau mày.</div>
</div>

<p>Lật bờ dưới cơ cau mày lên để làm lộ nguyên ủy của thân dây thần kinh trên ổ mắt cùng các nhánh nông và sâu của nó. Các nhánh thần kinh liên quan đến cơ cau mày được ký hiệu là: <strong>SON-S<sub>CSM</sub></strong> (nhánh từ thần kinh trên ổ mắt nông đến cơ cau mày) hoặc <strong>SON-D<sub>CSM</sub></strong> (nhánh từ thần kinh trên ổ mắt sâu đến cơ cau mày).</p>

<h2 class="section-heading">KẾT QUẢ</h2>

<p class="no-indent">Phát hiện 4 dạng phân nhánh của dây thần kinh trên ổ mắt tương quan với các sợi cơ cau mày (Bảng 1):</p>

<div class="table-box">
  <div class="table-title">Bảng 1. Phân loại các dạng phân nhánh Thần kinh trên ổ mắt</div>
  <table>
    <thead>
      <tr>
        <th>Phân loại</th>
        <th>Nguồn gốc nhánh</th>
        <th>Phải</th>
        <th>Trái</th>
        <th>Tổng thể (%)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Type I</strong></td>
        <td>Nhánh sâu (SON-D)</td>
        <td>10</td>
        <td>9</td>
        <td>20/50 (40%)</td>
      </tr>
      <tr>
        <td><strong>Type II</strong></td>
        <td>Nhánh sâu & nhánh nông</td>
        <td>9</td>
        <td>8</td>
        <td>17/50 (34%)</td>
      </tr>
      <tr>
        <td><strong>Type III</strong></td>
        <td>Nhánh nông, không có nhánh sâu</td>
        <td>1</td>
        <td>1</td>
        <td>2/50 (4%)</td>
      </tr>
      <tr>
        <td><strong>Type IV*</strong></td>
        <td>Không liên quan đến cơ cau mày</td>
        <td>5</td>
        <td>5</td>
        <td>11/50 (22%)</td>
      </tr>
    </tbody>
  </table>
  <div class="table-footnote">*Phân nhánh thường bắt đầu ở vị trí cao hơn về phía đầu so với khối cơ cau mày.</div>
</div>

<p>• <strong>Type I (40%):</strong> Nhánh sâu cho một nhánh duy nhất chạy trực tiếp dọc mặt dưới cơ cau mày (Hình 3).</p>

<div class="figure-box">
  <img src="${p2ImgDir}/fig3.jpg" alt="Hình 3">
  <div class="figure-caption"><strong>Hình 3.</strong> Dạng phân nhánh thần kinh trên ổ mắt Type I. SON-D: nhánh sâu; SON-D<sub>CSM</sub>: nhánh từ nhánh sâu đi vào cơ; CSM: cơ cau mày. (Phía dưới) Sơ đồ minh họa đơn giản hóa.</div>
</div>

<p>• <strong>Type II (34%):</strong> Quan sát thấy các nhánh từ nhánh nông (SON-S) bên cạnh các sợi từ nhánh sâu (Hình 4).</p>

<div class="figure-box">
  <img src="${p2ImgDir}/fig4.jpg" alt="Hình 4">
  <div class="figure-caption"><strong>Hình 4.</strong> Dạng phân nhánh thần kinh trên ổ mắt Type II. SON-S: nhánh nông; SON-S<sub>CSM</sub>: sợi nhánh từ nhánh nông đến cơ cau mày; CSM: cơ cau mày; SON-D: nhánh sâu; SON-D<sub>CSM</sub>: nhánh từ nhánh sâu.</div>
</div>

<p>• <strong>Type III (4%):</strong> Có các nhánh riêng biệt từ nhánh nông (SON-S) đi vào cơ cau mày, nhưng không có nhánh từ nhánh sâu (Hình 5).</p>

<div class="figure-box">
  <img src="${p2ImgDir}/fig5.jpg" alt="Hình 5">
  <div class="figure-caption"><strong>Hình 5.</strong> Dạng phân nhánh thần kinh trên ổ mắt Type III. SON-S: nhánh nông; SON-S<sub>CSM</sub>: nhánh từ nhánh nông vào cơ; CSM: cơ cau mày; SON-D: nhánh sâu.</div>
</div>

<p>• <strong>Type IV (22%):</strong> Không có nhánh thần kinh nào phân bố trực tiếp vào khối cơ cau mày (Hình 6). Phân nhánh chỉ bắt đầu sau khi thần kinh đã vượt qua bờ trên của cơ cau mày.</p>

<div class="figure-box">
  <img src="${p2ImgDir}/fig6.jpg" alt="Hình 6">
  <div class="figure-caption"><strong>Hình 6.</strong> Dạng phân nhánh thần kinh trên ổ mắt Type IV. (Trên) Góc nhìn từ trên xuống dưới vào cơ cau mày đã được kéo vén. (Dưới) Sơ đồ minh họa phân nhánh xuất phát cao hơn hẳn so với cơ cau mày.</div>
</div>

<p>Khi có khuyết xương trên ổ mắt, thỉnh thoảng xuất hiện một <em>dải xơ căng (taut fibrous band)</em> tạo thành bờ trước của khuyết và ép chặt thân thần kinh vào xương trán.</p>

<div class="figure-box">
  <img src="${p2ImgDir}/fig7.jpg" alt="Hình 7">
  <div class="figure-caption"><strong>Hình 7.</strong> Tổng hợp phân loại các dạng phân nhánh thần kinh trên ổ mắt. (Trái) Dạng Type I phổ biến nhất (phóng to). (Phải) Các dạng Type II đến Type IV.</div>
</div>

<h2 class="section-heading">BÀN LUẬN</h2>

<p class="no-indent">Hiểu biết chính xác đường đi của dây thần kinh trên ổ mắt dọc theo cơ cau mày giúp nâng cao độ an toàn khi phẫu thuật qua đường mổ coronal, transpalpebral hay nội soi. Ở 74% trường hợp (Type I và II), nhánh thần kinh liên quan đến cơ cau mày bắt nguồn từ nhánh sâu (SON-D), trong khi ở 22% (Type IV) không có mối liên quan trực tiếp nào giữa thần kinh và cơ.</p>

<p>Các vị trí có nguy cơ chèn ép thần kinh trên ổ mắt tiềm tàng bao gồm: (1) Dải xơ căng ở bờ trước khuyết trên ổ mắt; (2) Vị trí nhánh sâu đổi hướng từ mặt phẳng bám sát màng xương chuyển sang mặt phẳng nông hơn để đi vào sợi cơ cau mày; (3) Vị trí đan xen của cơ cau mày với cơ trán và cơ vòng mi, nơi các lực cơ đối kháng vuông góc có thể tạo ra lực xoắn vặn (torqueing forces) chèn ép lên sợi thần kinh.</p>

<h2 class="section-heading">KẾT LUẬN</h2>

<p class="no-indent">Từ Phần II của nghiên cứu, chúng tôi rút ra các kết luận: (1) Sự chèn ép thần kinh trên ổ mắt trong căn nguyên đau nửa đầu xảy ra tại các điểm giải phẫu xác định dọc đường đi của thần kinh liên quan đến cơ cau mày; (2) Ở 78% trường hợp (Type I, II, III), thần kinh phân nhánh sớm gần nguyên ủy xương trán và đan xen mật thiết với cơ cau mày; (3) Tỷ lệ 78% này tương ứng rất chặt chẽ với tỷ lệ 79,5% bệnh nhân khỏi hoặc thuyên giảm đáng kể đau nửa đầu sau phẫu thuật cắt cơ cau mày của Guyuron và CS; (4) Những bệnh nhân không đáp ứng với phẫu thuật có thể thuộc nhóm Type IV (22%), nơi thần kinh không nằm trong khối cơ cau mày.</p>

<div class="citation-block">
  <strong>Liên hệ tác giả:</strong> Jeffrey E. Janis, M.D., Department of Plastic Surgery, University of Texas Southwestern Medical Center, 1801 Inwood Road, Dallas, Texas 75390-9132.<br>
  <strong>Email:</strong> Jeffrey.Janis@utsouthwestern.edu<br>
  <strong>Lời cảm ơn:</strong> Các tác giả trân trọng cảm ơn BS. Rod J. Rohrich đã tài trợ mẫu xác tươi, và Kim A. Hoggatt-Krumwiede cùng Holly Smith đã hỗ trợ thực hiện các hình ảnh minh họa y khoa trong nghiên cứu này.
</div>

</div>

</body>
</html>
`;

fs.writeFileSync(path.join(baseDir, 'part1_vietnamese.html'), part1Html, 'utf8');
fs.writeFileSync(path.join(baseDir, 'part2_vietnamese.html'), part2Html, 'utf8');
console.log('HTML files generated successfully!');

(async () => {
  const browser = await chromium.launch();
  
  // Render Part 1 PDF
  const page1 = await browser.newPage();
  await page1.goto('file://' + path.join(baseDir, 'part1_vietnamese.html'), { waitUntil: 'networkidle' });
  const pdf1Out = path.join(baseDir, 'Giai_Phau_Co_Cau_May_Phan_1_Dia_Hinh_Co.pdf');
  await page1.pdf({
    path: pdf1Out,
    format: 'A4',
    printBackground: true,
    margin: { top: '15mm', bottom: '15mm', left: '15mm', right: '15mm' },
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size:8pt; font-family:Times New Roman,serif; color:#777; width:100%; text-align:left; padding-left:15mm; font-style:italic;">Tạp chí Phẫu thuật Tạo hình & Tái tạo • Bản dịch tiếng Việt</div>',
    footerTemplate: '<div style="font-size:8pt; font-family:Times New Roman,serif; color:#555; width:100%; text-align:center;">Trang <span class="pageNumber"></span> / <span class="totalPages"></span></div>'
  });
  console.log('PDF 1 created:', pdf1Out);

  // Render Part 2 PDF
  const page2 = await browser.newPage();
  await page2.goto('file://' + path.join(baseDir, 'part2_vietnamese.html'), { waitUntil: 'networkidle' });
  const pdf2Out = path.join(baseDir, 'Giai_Phau_Co_Cau_May_Phan_2_Phan_Nhanh_TK_Tren_O_Mat.pdf');
  await page2.pdf({
    path: pdf2Out,
    format: 'A4',
    printBackground: true,
    margin: { top: '15mm', bottom: '15mm', left: '15mm', right: '15mm' },
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size:8pt; font-family:Times New Roman,serif; color:#777; width:100%; text-align:left; padding-left:15mm; font-style:italic;">Tạp chí Phẫu thuật Tạo hình & Tái tạo • Bản dịch tiếng Việt</div>',
    footerTemplate: '<div style="font-size:8pt; font-family:Times New Roman,serif; color:#555; width:100%; text-align:center;">Trang <span class="pageNumber"></span> / <span class="totalPages"></span></div>'
  });
  console.log('PDF 2 created:', pdf2Out);

  await browser.close();
  console.log('ALL PDF GENERATION COMPLETE!');
})();

