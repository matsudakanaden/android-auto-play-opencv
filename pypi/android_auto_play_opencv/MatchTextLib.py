import cv2 # pip install opencv-python
import numpy
import easyocr

class MatchTextLib():
    
    img = None
    temp = None
    loc = []
    minVal = 0
    maxVal = 0
    minLoc = []
    maxLoc = []
    
    #類似度の設定(0~1)
    THRESHOLD = 0.8

    def matchText(self, _img, _temp, _screenshot = None, _threshold = None):
        """
        テキストマッチング
        
        Parameters:
        ----------
        _img : Bytes
            Image byte array
        _temp : str
            Text to search
        _screenshot : str | None
            Base image path
        _threshold : float | None
            Threshold for whether text are included
        """

        if _threshold is None:
            _threshold = self.THRESHOLD

        if (_img is None or len(_img) == 0) and _screenshot is None:
            raise Exception('スクリーンショットが撮れていません。screencap を実行してから処理を開始して下さい。')

        if _screenshot is not None:
            self.img = cv2.imread(_screenshot, 0)
        else:
            self.img = cv2.imdecode(numpy.frombuffer(_img, numpy.uint8), 0)
            
        #画像の文字を抽出
        reader = easyocr.Reader(['ja'])
        result = reader.readtext(_screenshot)
        for text in result:
            if text ==　_temp:
                self.milLoc =  text[0][0],text[0][1]
                self.maxLoc = text[0][2],text[0][3]

                
                
                
            print(text)
        
        
        # テンプレートマッチング
        match_result = cv2.matchTemplate(self.img, self.temp, cv2.TM_CCOEFF_NORMED)

        #検出結果から検出領域の位置を取得
        self.loc = numpy.where(match_result >= _threshold)
        
        # 最も類似度が高い位置と低い位置を取得します
        self.minVal, self.maxVal, self.minLoc, self.maxLoc = cv2.minMaxLoc(match_result)

    def judgeMatching(self, _threshold = None):
        """
        テンプレートマッチング結果に閾値を超えてる箇所があるか
        
        Parameters:
        ----------
        _threshold : float | None
            Threshold for whether template images are included
        """
        
        if _threshold is None:
            _threshold = self.THRESHOLD

        if _threshold < self.maxVal:
            return True
        else:
            return False

    def getCenterPos(self):
        w, h = self.temp.shape[::-1]
        for top_left in zip(*self.loc[::-1]):
            return (top_left[0] + (w /2), top_left[1] + (h / 2))

    def getCenterPosMulti(self):
        w, h = self.temp.shape[::-1]
        result: tuple = []
        for top_left in zip(*self.loc[::-1]):
            is_NG = False
            tmp_pos = (top_left[0] + (w /2), top_left[1] + (h / 2))

            # 近い場所が既に存在しているかの確認
            for tmp_res in result:
                if tmp_res[0] - (w /2) < tmp_pos[0] < tmp_res[0] + (w /2) and \
                    tmp_res[1] - (h / 2) < tmp_pos[1] < tmp_res[1] + (h / 2):
                    is_NG = True
                    break
                pass
            
            if is_NG == False:
                result.append(tmp_pos)
                pass
        
        return result

    def rawToOpenCVImg(self, raw):
        # wigth, heightを取得
        wigth = int.from_bytes(raw[0:4], 'little')
        height = int.from_bytes(raw[4:8], 'little')
        _ = int.from_bytes(raw[8:12], 'little')

        # ここのCopyは必須。そうでないと、編集が出来ない
        tmp = numpy.frombuffer(raw[12:], numpy.uint8, -1, 0).copy() 

        # 配列の形状変換。
        # 1つの要素がRGBAである、height * widthの行列を作る。
        img = numpy.reshape(tmp, (height, wigth, 4))    

        # 要素入れ替え。
        # RawDataはRGB、OpenCVはBGRなので、0番目の要素と、2番目の要素を入れ替える必要がある。
        b = img[:, :, 0].copy()               # ここのコピーも必須
        img[:, :, 0] = img[:, :, 2] 
        img[:, :, 2] = b

        # alpha値を削除。alpha値が必要な場合は、下記行は消しても良いかも？
        img2 = numpy.delete(img, 3, 2)

        return img2
