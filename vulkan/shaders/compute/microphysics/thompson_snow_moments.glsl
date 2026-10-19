vec2 snow_polynomial_a(vec2 t, vec2 m)
{
    vec2 t2 = df_mul(t, t), m2 = df_mul(m, m);
    return df_sub(
        df_add(
            df_add(df_add(df_sub(df_add(df_sub(df_sub(vec2(5.0653390884399414, -8.8439941592355353e-08),
                                                      df_mul(vec2(0.062659002840518951, -2.8405189450575463e-09), t)),
                                               df_mul(vec2(3.0323619842529297, 1.5747070314375833e-08), m)),
                                        df_mul(df_mul(vec2(0.029469000175595284, -1.7559528484034637e-10), t), m)),
                                 df_mul(vec2(0.00028499998734332621, 1.2656673777609873e-11), t2)),
                          df_mul(vec2(0.31255000829696655, -8.2969665582410812e-09), m2)),
                   df_mul(df_mul(vec2(0.00020399999630171806, 3.6982819421899765e-12), t2), m)),
            df_mul(df_mul(vec2(0.0031989999115467072, 8.845329286730208e-11), t), m2)),
        df_mul(df_mul(vec2(0.015952000394463539, -3.9446353822247815e-10), m2), m));
}
vec2 snow_polynomial_b(vec2 t, vec2 m)
{
    vec2 t2 = df_mul(t, t), m2 = df_mul(m, m);
    return df_sub(
        df_add(df_add(df_add(df_sub(df_add(df_add(df_sub(vec2(0.4762209951877594, 4.8122406060002731e-09),
                                                         df_mul(vec2(0.01589599996805191, 3.194809000683918e-11), t)),
                                                  df_mul(vec2(0.16597700119018555, -1.1901855334617295e-09), m)),
                                           df_mul(df_mul(vec2(0.007468000054359436, -5.4359436223172519e-11), t), m)),
                                    df_mul(vec2(0.00014099999680183828, 3.1981617332720036e-12), t2)),
                             df_mul(vec2(0.060366000980138779, -9.8013877569691488e-10), m2)),
                      df_mul(df_mul(vec2(7.8999997640494257e-05, 2.3595057385781409e-12), t2), m)),
               df_mul(df_mul(vec2(0.00059399998281151056, 1.7188489455539913e-11), t), m2)),
        df_mul(df_mul(vec2(0.0035769999958574772, 4.1425226963986983e-12), m2), m));
}
vec2 snow_moment(vec2 second, vec2 t, vec2 order)
{
    vec2 a = snow_polynomial_a(t, order), b = snow_polynomial_b(t, order);
    return df_mul(df_real_power(vec2(10, 0), a), df_real_power(second, b));
}
