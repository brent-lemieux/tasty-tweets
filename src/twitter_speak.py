'''Create and pickle a dictionary that maps twitter
slang to "proper" english'''

import pickle

twitter_speak = {' af ':' as fuck ',
                ' rn':' right now ',
                ' lol ':' laugh out loud ',
                ' lmao ':' laughing my ass off ',
                ' bruh ':' brother ',
                ' bro ':' brother ',
                ' rly ':' really ',
                ' (: ':' happy ',
                ' :( ':' sad ',
                ' smh ':' shaking my head ',
                ' smdh ':' shaking my damn head ',
                ' guac ':' guacamole ',
                ' ok ': ' okay ',
                ' lit ':' awesome ',
                ' dope ':' awesome ',
                ' chill ': ' awesome ',
                ' imo ':' in my opinion ',
                ' dm ':' message ',
                ' fyi ':' for your information ',
                ' fym ':' free your mind ',
                ' ur ':' you are ',
                ' omg ':' oh my god ',
                ' finna ':' going to ',
                ' gonna ':' going to ',
                ' idc ':' i dont care ',
                ' idk ':' i dont know ',
                ' wtf ':' what the fuck ',
                ' pls ':' please ',
                ' hearteyes ':' heart ',
                ' yellowheart ':' heart ',
                ' blueheart ':' heart ',
                ' purpleheart ':' heart ',
                ' greenheart ':' heart ',
                ' twohearts ':' heart ',
                ' sparklingheart ':' heart ',
                ' heartpulse ':' heart ',
                ' kissingheart ':' heart ',
                ' tryna ':' trying ',
                ' som1 ':' someone ',
                ' tf ':' the fuck '
                }

pickle.dump( twitter_speak, open( 'twitter_speak.pkl', 'wb' ) )
