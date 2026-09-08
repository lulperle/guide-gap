<!--
Raw generated output, committed unedited, from the run recorded in
evals/results.json (t14 + t15). Nobody reviewed or corrected it. It is here as
evidence of what the drafting prompt produces -- specifically that 確認が必要な点
names the tag naming rules as absent from the source material instead of inventing
them, which is the behaviour the prompt exists to enforce. It is not a guide page
and it is not accurate.
-->

# タグの名前は何にすればいい？費用按分用ラベルの命名規則について

## どんなときに読むページか

- 複数課で1つの環境を共用しており、費用を課ごとに按分したい
- 按分にはリソースへのタグ付けが必要と聞いたが、タグの名前をどう決めればよいかわからない
- タグの命名規則が団体側で自由に決めてよいのか、全体で統一されたルールがあるのか知りたい

## 手順

1. 複数課で1つの環境を共用している場合、課ごとの按分はリソースに付与されたタグを使って集計されます。
2. タグが付いていないリソースの費用は「未分類」として出力され、按分の対象外になります。
3. 按分結果を確認するには、利用者ポータルの「コスト」画面右上の「エクスポート」から対象年月を指定し、CSV を出力します。
4. 出力された CSV で「未分類」の金額を確認してください。想定と大きく異なる場合、タグの付与漏れによって別の課へ集計されているか、未分類に落ちているケースが多いです。
5. それでも説明がつかない場合は、CSV を添付してサービスデスクへ照会してください。

## 確認が必要な点

- タグの名前（キー・値）に決められた命名規則があるかどうかは、既存の資料からは確認できませんでした。
- タグの名前を団体・課ごとに自由に決めてよいのか、全体で統一された付け方があるのかについても、資料に記載がありませんでした。
- タグを実際にリソースへ付与する操作手順（どの画面で、どのように設定するか）についても記載がなく、確認が必要です。
- これらの点は、サービスデスクへ直接問い合わせて確認することをお勧めします。
