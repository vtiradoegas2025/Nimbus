#include "thompson_compensated.glsl"
#include "thompson_number_math.glsl"
bool  failed;
vec2  q[7];
vec2  original_q[7];
float original_n[5];
float rates[5];
vec2  t, p, rho, step, heat, snow_vapor;
float snow_speed;
float ice_index, rs_index, rg_index;
bool  finite_pair(vec2 x)
{
    return !isnan(x.x) && !isinf(x.x) && !isnan(x.y) && !isinf(x.y);
}
vec2 df_min(vec2 a, vec2 b)
{
    if (!finite_pair(a) || !finite_pair(b))
        failed = true;
    return df_sub(a, b).x < 0.0 ? a : b;
}
vec2 df_max(vec2 a, vec2 b)
{
    if (!finite_pair(a) || !finite_pair(b))
        failed = true;
    return df_sub(a, b).x > 0.0 ? a : b;
}
vec2 df_power(vec2 x, int n)
{
    vec2 v = vec2(1, 0);
    for (int i = 0; i < n; ++i)
        v = df_mul(v, x);
    return v;
}
vec2 logarithm(vec2 x)
{
    if (!finite_pair(x) || x.x <= 0.0)
    {
        failed = true;
        return vec2(0);
    }
    return df_log(x);
}
vec2 exponential(vec2 x)
{
    vec2 v = df_exp(x);
    if (!finite_pair(v))
        failed = true;
    return v;
}
vec2 root(vec2 x)
{
    if (!finite_pair(x) || x.x < 0.0)
    {
        failed = true;
        return vec2(0);
    }
    return x.x == 0.0 ? vec2(0) : df_sqrt(x);
}
vec2 root3(vec2 x)
{
    if (!finite_pair(x) || x.x < 0.0)
    {
        failed = true;
        return vec2(0);
    }
    return df_cbrt(x);
}
vec2 df_real_power(vec2 x, vec2 power)
{
    if (x.x == 0.0 && power.x > 0.0)
        return vec2(0);
    return exponential(df_mul(power, logarithm(x)));
}
vec2 coeff(int index)
{
    return vec2(coefficients[2 * index], coefficients[2 * index + 1]);
}
#include "thompson_snow_moments.glsl"
vec2 current_number(int i)
{
    return df_max(vec2(0), df_add(vec2(original_n[i], 0), df_mul(step, vec2(rates[i], 0))));
}
void store_number(int i, vec2 candidate)
{
    rates[i]      = stored_tendency(original_n[i], df_max(vec2(0), candidate));
    vec2 residual = df_add(vec2(original_n[i], 0), df_mul(step, vec2(rates[i], 0)));
    if (dt_low != 0.0 && rates[i] < 0.0 && residual.x <= original_n[i] * 1e-13)
        rates[i] = uintBitsToFloat(floatBitsToUint(rates[i]) - 1u);
}
void add_number(int i, vec2 rate)
{
    rates[i] += scalar(rate);
}
void transfer_mass(int source, int target, vec2 amount, vec2 latent)
{
    amount = df_min(q[source], amount);
    if (amount.x <= 0.0)
        return;
    q[source] = df_sub(q[source], amount);
    q[target] = df_add(q[target], amount);
    heat      = df_add(heat, df_mul(latent, amount));
}
int axis_index(vec2 v, int base, int count)
{
    if (!finite_pair(v))
    {
        failed = true;
        return 0;
    }
    int low = 0, high = count;
    while (high - low > 1)
    {
        int mid = (low + high) / 2;
        if (df_sub(v, coeff(base + mid)).x >= 0.0)
            low = mid;
        else
            high = mid;
    }
    return low;
}
vec2 buck(vec2 temp, vec2 press, bool ice)
{
    vec2 tc = df_sub(df_min(vec2(400.0, 0.0), df_max(vec2(150.0, 0.0), temp)),
                     vec2(273.14999389648438, 6.1035156022626325e-06));
    vec2 e =
        ice ? df_mul(vec2(611.1500244140625, -2.4414062522737368e-05),
                     exponential(df_div(df_mul(df_sub(vec2(23.035999298095703, 7.0190429823924205e-07),
                                                      df_div(tc, vec2(333.70001220703125, -1.2207031261368684e-05))),
                                               tc),
                                        df_add(vec2(279.82000732421875, -7.3242187568212103e-06), tc))))
            : df_mul(vec2(611.21002197265625, -2.1972656213620212e-05),
                     exponential(df_div(
                         df_mul(df_sub(vec2(18.677999496459961, 5.0354003988672957e-07), df_div(tc, vec2(234.5, 0.0))),
                                tc),
                         df_add(vec2(257.1400146484375, -1.4648437513642421e-05), tc))));
    return df_min(vec2(0.20000000298023224, -2.980232227667301e-09),
                  df_max(vec2(0.0, 0.0),
                         df_div(df_mul(vec2(0.62198001146316528, -1.1463165305869438e-08), e),
                                df_max(df_sub(df_min(vec2(200000.0, 0.0), df_max(vec2(100.0, 0.0), press)), e),
                                       vec2(1.0, 0.0)))));
}
vec2 liquid_sat(vec2 press, vec2 temp)
{
    vec2 x = df_max(-vec2(80.0, 0.0), df_sub(temp, vec2(273.16000366210938, -3.6621093499888957e-06)));
    vec2 e = -vec2(3.2158239423154059e-14, -1.2315406151497315e-22);
    e      = df_add(vec2(3.795342946172342e-12, 1.5382765800993744e-19), df_mul(x, e));
    e      = df_add(vec2(7.0262071716342689e-09, -1.9163426923402981e-16), df_mul(x, e));
    e      = df_add(vec2(2.0315417259553215e-06, 9.4044678576094996e-14), df_mul(x, e));
    e      = df_add(vec2(0.00029929107404313982, 6.956860163897377e-12), df_mul(x, e));
    e      = df_add(vec2(0.026422431692481041, 4.0751895954982409e-10), df_mul(x, e));
    e      = df_add(vec2(1.4317715167999268, 5.3200073235615264e-08), df_mul(x, e));
    e      = df_add(vec2(44.460689544677734, 5.532226765581072e-08), df_mul(x, e));
    e      = df_add(vec2(611.58367919921875, 1.9800781274170731e-05), df_mul(x, e));
    e      = df_min(e, df_mul(press, vec2(0.15000000596046448, -5.9604644830901776e-09)));
    return df_div(df_mul(vec2(0.62199997901916504, 2.0980834958272965e-08), e), df_sub(press, e));
}
vec2 theta_e(vec2 temp, vec2 vapor, vec2 lcl)
{
    vec2 adjusted = df_add(vapor, vec2(9.9999999392252903e-09, 6.0774709918447105e-17));
    return df_mul(
        df_mul(temp,
               df_real_power(df_div(vec2(100000.0, 0.0), p),
                             df_mul(vec2(0.28540000319480896, -3.1948089729283424e-09),
                                    df_sub(vec2(1.0, 0.0),
                                           df_mul(vec2(0.2800000011920929, -1.1920928688624599e-09), adjusted))))),
        exponential(
            df_mul(df_mul(df_sub(df_div(vec2(3.375999927520752, 7.2479247936740876e-08), lcl),
                                 vec2(0.0025400000158697367, -1.5869736514489974e-11)),
                          df_mul(adjusted, vec2(1000.0, 0.0))),
                   df_add(vec2(1.0, 0.0), df_mul(vec2(0.81000000238418579, -2.3841857377249198e-09), adjusted)))));
}
vec2 wet_bulb(vec2 vapor)
{
    vec2 sat = liquid_sat(p, t);
    if (df_sub(vapor, df_mul(vec2(0.99900001287460327, -1.2874603272372553e-08), sat)).x >= 0.0)
        return t;
    vec2 adjusted = df_add(vapor, vec2(9.9999999392252903e-09, 6.0774709918447105e-17));
    vec2 l =
        logarithm(df_div(df_mul(p, adjusted), df_add(vec2(0.62199997901916504, 2.0980834958272965e-08), adjusted)));
    vec2 dew    = df_min(df_sub(t, vec2(0.0010000000474974513, -4.7497451284572723e-11)),
                         df_div(df_sub(df_mul(vec2(35.860000610351562, -6.1035156306843419e-07), l),
                                       vec2(4947.232421875, 7.8125000072759576e-05)),
                                df_sub(l, vec2(23.683700561523438, -5.615234357492227e-07))));
    vec2 lcl    = df_add(df_div(vec2(1.0, 0.0),
                                df_add(df_div(vec2(1.0, 0.0), df_sub(dew, vec2(56.0, 0.0))),
                                       df_div(logarithm(df_div(t, dew)), vec2(800.0, 0.0)))),
                         vec2(56.0, 0.0));
    vec2 target = theta_e(t, vapor, lcl);
    vec2 guess =
        df_mul(df_sub(target,
                      df_mul(vec2(0.5, 0.0),
                             df_real_power(df_max(df_sub(target, vec2(270.0, 0.0)), vec2(0.0, 0.0)),
                                           vec2(1.0499999523162842, 4.7683715864721421e-08)))),
               df_real_power(df_div(p, vec2(100000.0, 0.0)), vec2(0.20000000298023224, -2.980232227667301e-09)));
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        if (!finite_pair(guess) || guess.x <= 0.0)
        {
            failed = true;
            return t;
        }
        vec2 next = df_add(guess, vec2(1.0, 0.0));
        vec2 a = theta_e(guess, liquid_sat(p, guess), guess), b = theta_e(next, liquid_sat(p, next), next);
        vec2 denominator = df_sub(b, a);
        if (!finite_pair(denominator) || denominator.x == 0.0)
            break;
        vec2 correction = df_div(df_sub(target, a), denominator);
        if (!finite_pair(correction))
            break;
        guess = df_add(guess, correction);
        if (abs(correction.x) < .01)
            return df_min(t, guess);
    }
    vec2 x = df_min(vec2(475.0, 0.0), target), result;
    if (df_sub(x, vec2(335.5, 0.0)).x <= 0.0)
        result = df_add(
            df_add(
                -vec2(1.0092229135727493e-10, 6.4272507451920209e-19),
                df_mul(
                    x,
                    df_add(
                        -vec2(1.4794534486384237e-08, -8.6384236572339868e-17),
                        df_mul(
                            x,
                            df_add(
                                -vec2(1.730375743136392e-06, -4.3136392048127078e-14),
                                df_mul(
                                    x,
                                    df_add(-vec2(0.00012708999565802515, 4.3419748590838025e-12),
                                           df_mul(x,
                                                  df_add(vec2(1.1584986623347504e-06, 7.6652496436497349e-15),
                                                         df_mul(x,
                                                                df_add(-vec2(3.5182967650371211e-09,
                                                                             9.5962878877863072e-17),
                                                                       df_mul(x,
                                                                              vec2(3.5741522588783603e-12,
                                                                                   -5.887836016845827e-20))))))))))))),
            vec2(273.14999389648438, 6.1035156022626325e-06));
    else
        result = df_add(
            df_mul(
                x,
                df_add(
                    -vec2(3.5223512995230521e-10, 4.7694768830713015e-20),
                    df_mul(
                        x,
                        df_add(
                            -vec2(5.7250808538356068e-08, -1.5383560660428591e-15),
                            df_mul(x,
                                   df_add(-vec2(5.839754066983005e-06, 1.5301699489466466e-13),
                                          df_mul(x,
                                                 df_add(vec2(4.7244515855027203e-08, 4.4497279756294686e-16),
                                                        df_mul(x,
                                                               df_add(-vec2(1.1340284267191691e-10,
                                                                            2.3280830836230922e-18),
                                                                      df_mul(x,
                                                                             vec2(8.7295805078959482e-14,
                                                                                  -1.0589594828088389e-21)))))))))))),
            vec2(273.14999389648438, 6.1035156022626325e-06));
    return df_min(
        t,
        df_mul(result,
               df_real_power(df_div(p, vec2(100000.0, 0.0)), vec2(0.28600001335144043, -1.3351440453224228e-08))));
}
void nucleation()
{
    vec2 vapor = q[0], nw = current_number(3), na = current_number(4), ni = current_number(2), snow = q[4];
    vec2 satw = buck(t, p, false), sati = buck(t, p, true);
    vec2 sw = df_sub(df_div(vapor, satw), vec2(1.0, 0.0)), si = df_sub(df_div(vapor, sati), vec2(1.0, 0.0)),
         nic = df_mul(ni, rho), nwc = df_mul(nw, rho), nac = df_mul(na, rho);
    vec2 demott = vec2(0), koop = vec2(0), tc = df_sub(t, vec2(273.14999389648438, 6.1035156022626325e-06));
    if (df_sub(si, vec2(0.25, 0.0)).x >= 0.0 ||
        (df_sub(sw, vec2(1.0000000036274937e-15, -3.627493647833322e-24)).x > 0.0 &&
         df_sub(t, vec2(253.14999389648438, 6.1035156306843419e-06)).x < 0.0))
    {
        vec2 reference = df_div(
            vec2(101325.0, 0.0),
            df_mul(vec2(287.04998779296875, 1.2207031261368684e-05), vec2(273.14999389648438, 6.1035156022626325e-06)));
        vec2 aerosol =
            df_max(vec2(0.5, 0.0),
                   df_div(df_mul(df_mul(nac, reference), vec2(9.9999999747524271e-07, 2.5247572468697606e-15)), rho));
        vec2 target = df_mul(
            df_div(df_mul(df_mul(df_mul(vec2(5.9400001191534102e-05, -1.1915341014746318e-12),
                                        df_real_power(-tc, vec2(3.3299999237060547, 7.6293945383554274e-08))),
                                 df_real_power(aerosol,
                                               df_add(df_mul(-vec2(0.026399999856948853, 1.4305114731660851e-10), tc),
                                                      vec2(0.0032999999821186066, 1.7881393414576063e-11)))),
                          rho),
                   reference),
            vec2(1000.0, 0.0));
        demott = df_min(nac, df_max(vec2(0.0, 0.0), df_sub(target, nic)));
    }
    vec2 snow_count =
        df_sub(snow, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x <= 0.0 ? vec2(0) : vec2(1000000.0, 0.0);
    if (df_sub(df_add(df_add(snow_count, nic), demott), vec2(999000.0, 0.0)).x <= 0.0 &&
        df_sub(t, vec2(238.0, 0.0)).x < 0.0 && df_sub(si, vec2(0.40000000596046448, -5.960464455334602e-09)).x >= 0.0)
    {
        vec2 chem =
            df_sub(df_sub(df_add(vec2(210368.0, 0.0), df_mul(vec2(131.43800354003906, -3.5400390743234311e-06), t)),
                          df_div(vec2(3323730.0, 0.0), t)),
                   df_mul(vec2(41729.1015625, -0.0015625000014551915), logarithm(t)));
        vec2 activity = exponential(df_div(chem, df_mul(vec2(8.314000129699707, -1.2969970697440658e-07), t)));
        vec2 delta    = df_sub(df_div(vapor, satw), activity);
        vec2 log_rate = df_min(
            vec2(20.0, 0.0),
            df_add(df_sub(df_add(-vec2(906.70001220703125, -1.2207031204525265e-05), df_mul(vec2(8502.0, 0.0), delta)),
                          df_mul(df_mul(vec2(26924.0, 0.0), delta), delta)),
                   df_mul(df_mul(df_mul(vec2(29180.0, 0.0), delta), delta), delta)));
        vec2 probability =
            df_min(df_sub(vec2(1.0, 0.0),
                          exponential(df_mul(
                              df_mul(-df_real_power(vec2(10.0, 0.0), log_rate),
                                     df_mul(df_mul(df_mul(df_mul(df_div(vec2(4.0, 0.0), vec2(3.0, 0.0)),
                                                                 vec2(3.1415927410125732, -8.7412573179790343e-08)),
                                                          vec2(2.4999999368446879e-06, 6.3155312295619772e-14)),
                                                   vec2(2.4999999368446879e-06, 6.3155312295619772e-14)),
                                            vec2(2.4999999368446879e-06, 6.3155312295619772e-14))),
                              step))),
                   vec2(1.0, 0.0));
        koop = df_max(vec2(0.0, 0.0), df_min(df_mul(probability, nwc), vec2(1000000.0, 0.0)));
    }
    vec2 mass   = df_add(df_mul(vec2(9.999999960041972e-13, 3.9958027796366288e-21), demott),
                         df_mul(vec2(9.9999998245167004e-14, 1.7548329986192532e-21), koop));
    vec2 factor = mass.x > 0.0 ? df_min(vec2(1.0, 0.0),
                                        df_div(df_max(vec2(0.0, 0.0),
                                                      df_mul(df_mul(df_sub(vapor, sati), rho),
                                                             vec2(0.99900001287460327, -1.2874603272372553e-08))),
                                               mass))
                               : vec2(0);
    demott      = df_mul(demott, factor);
    koop        = df_mul(koop, factor);
    mass        = df_div(df_add(df_mul(vec2(9.999999960041972e-13, 3.9958027796366288e-21), demott),
                                df_mul(vec2(9.9999998245167004e-14, 1.7548329986192532e-21), koop)),
                         rho);
    transfer_mass(0, 3, mass, df_div(vec2(2834000.0, 0.0), vec2(1004.0, 0.0)));
    add_number(2, df_div(df_div(df_add(demott, koop), rho), step));
    store_number(3, df_sub(nw, df_div(koop, rho)));
    store_number(4, df_sub(na, df_div(demott, rho)));
}
vec2 ice_sorted(vec2 mass, vec2 supplied)
{
    vec2 number = df_max(vec2(9.9999999747524271e-07, 2.5247572468697606e-15), supplied);
    vec2 c      = df_div(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(890.0, 0.0)), vec2(6.0, 0.0));
    if (df_sub(number, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x <= 0.0)
        return df_min(
            vec2(999000.0, 0.0),
            df_div(
                df_mul(mass, df_power(df_div(vec2(4.0, 0.0), vec2(4.9999998736893758e-06, 1.2631062459123954e-13)), 3)),
                df_mul(vec2(6.0, 0.0), c)));
    vec2 lam = root3(df_div(df_mul(df_mul(c, vec2(6.0, 0.0)), number), mass)), diameter = df_div(vec2(4.0, 0.0), lam);
    if (df_sub(diameter, vec2(4.9999998736893758e-06, 1.2631062459123954e-13)).x < 0.0)
        number = df_min(
            vec2(999000.0, 0.0),
            df_div(
                df_mul(mass, df_power(df_div(vec2(4.0, 0.0), vec2(4.9999998736893758e-06, 1.2631062459123954e-13)), 3)),
                df_mul(vec2(6.0, 0.0), c)));
    else if (df_sub(diameter, vec2(0.00030000001424923539, -1.4249235417897882e-11)).x > 0.0)
        number = df_div(
            df_mul(mass, df_power(df_div(vec2(4.0, 0.0), vec2(0.00030000001424923539, -1.4249235417897882e-11)), 3)),
            df_mul(vec2(6.0, 0.0), c));
    return number;
}
void ice_to_snow()
{
    vec2 mass = df_mul(q[3], rho), original = current_number(2), number = df_mul(original, rho);
    if (df_sub(mass, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x <= 0.0 ||
        df_sub(number, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x <= 0.0)
        return;
    vec2 sorted = ice_sorted(mass, number),
         lam    = root3(
             df_div(df_mul(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(890.0, 0.0)), sorted), mass)),
         diameter = df_div(vec2(4.0, 0.0), lam);
    int  im = axis_index(mass, 14862002, 64), inumber = axis_index(sorted, 14862066, 55);
    ice_index = float(inumber * 64 + im);
    vec2 mr = vec2(0), nr = vec2(0);
    if (im == 63 || df_sub(diameter, vec2(0.001500000013038516, -1.3038516013391677e-11)).x > 0.0)
    {
        mr = df_div(df_mul(vec2(0.99000000953674316, -9.5367431729442842e-09), mass), step);
        nr = df_div(df_mul(vec2(0.94999998807907104, 1.1920928910669204e-08), df_min(sorted, number)), step);
    }
    else if (df_sub(diameter, vec2(2.9999999242136255e-05, 7.5786374585337137e-13)).x >= 0.0)
    {
        vec2 wm = im < 63 ? df_min(vec2(1, 0), df_max(vec2(0), df_div(df_sub(mass, coeff(14862002 + im)),
            df_sub(coeff(14862003 + im), coeff(14862002 + im))))) : vec2(0);
        vec2 wn = inumber < 54 ? df_min(vec2(1, 0), df_max(vec2(0), df_div(df_sub(sorted, coeff(14862066 + inumber)),
            df_sub(coeff(14862067 + inumber), coeff(14862066 + inumber))))) : vec2(0);
        vec2 tm = vec2(0), tn = vec2(0);
        for (int ni = 0; ni < 2; ++ni)
            for (int mi = 0; mi < 2; ++mi)
            {
                vec2 weight = df_mul(ni == 0 ? df_sub(vec2(1, 0), wn) : wn,
                    mi == 0 ? df_sub(vec2(1, 0), wm) : wm);
                if (weight.x <= 0.0)
                    continue;
                int offset = 14841329 + 3 * ((inumber + ni) * 64 + im + mi);
                tm = df_add(tm, df_mul(weight, coeff(offset)));
                tn = df_add(tn, df_mul(weight, coeff(offset + 1)));
            }
        mr = df_min(df_div(df_mul(vec2(0.99000000953674316, -9.5367431729442842e-09), mass), step), df_div(tm, step));
        nr = df_min(df_div(df_mul(vec2(0.94999998807907104, 1.1920928910669204e-08), df_min(sorted, number)), step),
                    df_div(tn, step));
    }
    transfer_mass(3, 4, df_div(df_mul(step, mr), rho), vec2(0));
    store_number(2, df_sub(original, df_div(df_mul(step, nr), rho)));
}
