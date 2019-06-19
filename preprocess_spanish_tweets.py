# -*- coding: utf-8 -*-
import re, csv, copy
import string


def preprocess(tweet):

    tweet=tweet.decode('utf-8') #change coding if necessary!

    tweet=re.sub(re.escape(':('), r' tristeza ', tweet)
    tweet=re.sub(re.escape(':)'), r' reir ', tweet)
    tweet=re.sub(re.escape(':D'), r' reir ', tweet)
    tweet=re.sub(r'xD', r' reir ', tweet)
    tweet=re.sub(r'XD', r' reir ', tweet)
    tweet=re.sub(r'(ja){2,}', r' ja ', tweet)
    tweet=re.sub(r'(JA){2,}', r' ja ', tweet)
    tweet=re.sub('([.,!?()¡¿*&/|\:;"$%-=+–_])', r' \1 ', tweet)
    tweet=re.sub(ur'\u2014',ur' \u2014 ',tweet)
    tweet=tweet.split()
    new_tweet=''
    
    for wd in tweet:
        
        a=copy.copy(wd)

        emph_1 = re.findall(r'(([a-zA-Z])\2{2,})', wd)
        
        if len(emph_1)>0:
            new_tweet+='EMPHASIS '
            for x in emph_1:
                wd = wd.replace(x[0], x[1])
            

        if wd.startswith('#') or wd.startswith('@'):

            new_tweet+=wd[0]+' '
            
            wd=wd[1:]

            sort_e = re.findall(r'[A-Z]{4,}', wd)
            if len(sort_e)>0:
                for x in sort_e:
                    wd=wd[0]+wd[1:].lower()
                    
            sort_p = re.findall(r'[^a-zA-Z]', wd)
            if len(sort_p)>0:
                for x in list(set(sort_p)):
                    wd=re.sub(x, ' '+x+' ', wd)

            sort_l = re.findall(r'[a-z][A-Z][a-z]', wd)
            if len(sort_l)>0:
                for x in sort_l:
                    wd=re.sub(x, x[0]+' '+x[1:], wd)

            sort_c = re.findall(r'[A-Z][a-z][A-Z]', wd)
            if len(sort_c)>0:
                for x in sort_c:
                    wd=re.sub(x, x[:2]+' '+x[2], wd)
                    
        new_tweet+=wd+' '

    return new_tweet.encode('utf-8')

if __name__=='__main__':
    tweet = '"@marianorajoy: En España las cosas se pueden, se deben y se van a hacer infinitamente mejor que estos últimos 4 años" Eso son soluciones!!'
    print preprocess(tweet)

