float exact_divide(float a, float b) {
    uint ua=floatBitsToUint(a), ub=floatBitsToUint(b);
    uint sign=(ua^ub)&0x80000000u;
    uint aa=ua&0x7fffffffu, ab=ub&0x7fffffffu;
    uint ae=aa>>23, be=ab>>23;
    uint am=(aa&0x7fffffu)|(ae!=0u?0x800000u:0u);
    uint bm=(ab&0x7fffffu)|(be!=0u?0x800000u:0u);
    int ashift=23-max(findMSB(am),0), bshift=23-max(findMSB(bm),0);
    am<<=ashift;bm<<=bshift;
    int exponent=(max(int(ae),1)-ashift)-(max(int(be),1)-bshift);
    bool below=am<bm;
    exponent-=below?1:0;
    // Two guard bits and the exact integer remainder retain every discarded bit needed for ties-to-even.
    uint64_t numerator=uint64_t(am)<<(25+(below?1:0));
    uint64_t denominator=uint64_t(max(bm,1u));
    uint64_t quotient=numerator/denominator;
    uint q=uint(quotient), remainder=uint(numerator-quotient*denominator);
    // Additional shifts round subnormal results; the bound keeps even extreme underflow shifts defined.
    int shift=clamp(-126-exponent,0,27)+2;
    uint tail=q&((1u<<shift)-1u), halfway=1u<<(shift-1);
    uint rounded=q>>shift;
    rounded+=(tail>halfway||(tail==halfway&&(remainder!=0u||(rounded&1u)!=0u)))?1u:0u;
    bool carry=rounded>=0x1000000u;
    int out_exponent=max(exponent+127,0)+(carry?1:0);
    uint result=exponent< -126?rounded:((uint(out_exponent)<<23)|(rounded&0x7fffffu));
    if(out_exponent>=255) result=0x7f800000u;
    if(aa==0u||ab==0x7f800000u) result=0u;
    if(ab==0u||aa==0x7f800000u) result=0x7f800000u;
    if(aa>0x7f800000u||ab>0x7f800000u||(aa==0u&&ab==0u)||(aa==0x7f800000u&&ab==0x7f800000u)) result=0x7fc00000u;
    return uintBitsToFloat(sign|result);
}
