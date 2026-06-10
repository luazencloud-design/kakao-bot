// 파일명 기반 카테고리 자동 분류 (순수 함수).
// 우선순위: 오픈마켓(브랜드/가입안내) → 사업자등록 → 도구 → 강의자료 → 기타.
// "가입안내" 키워드가 도구명보다 우선 (예: 노션 가입안내서 → 오픈마켓).
export function inferCategory(filename: string): string {
  if (
    /(11번가|지마켓|gmarket|스마트스토어|쿠팡|coupang|롯데온|이베이|ebay|큐텐|qoo10|네이버쇼핑|카페24|옥션|auction|위메프|티몬|아마존|amazon|쇼피|shopee|lazada|라자다)/i.test(filename) ||
    /(가입안내|판매자|입점|셀러|seller|상품등록|계정연동)/i.test(filename)
  )
    return '오픈마켓가입';
  if (/(사업자|등록증|소명|세무|부가세|홈택스|hometax|세금계산서)/i.test(filename))
    return '사업자등록';
  if (/(노션|notion|niton|웨일|whale|zoom|slack)/i.test(filename)) return '도구가이드';
  if (
    /(강의|강좌|주차|수업|orientation|오리엔테이션|매출|수익|recording|transcript)/i.test(filename) ||
    /\.(vtt|mp4|mp3|m4a)$/i.test(filename)
  )
    return '강의자료';
  return '기타';
}
