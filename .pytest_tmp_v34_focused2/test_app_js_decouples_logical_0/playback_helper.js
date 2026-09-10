const P={time_s:[0,0.1,0.2,0.3,0.4]};const n=P.time_s.length;function frameAtOrBeforeTime(time) {
  if (time <= P.time_s[0]) return 0;
  if (time >= P.time_s[n-1]) return n-1;
  let low=0, high=n-1;
  while (low <= high) {
    const mid=(low+high)>>1;
    if (P.time_s[mid] <= time) low=mid+1; else high=mid-1;
  }
  return Math.max(0, high);
}
if(frameAtOrBeforeTime(.25)!==2||frameAtOrBeforeTime(.4)!==4||frameAtOrBeforeTime(9)!==4)process.exit(1);