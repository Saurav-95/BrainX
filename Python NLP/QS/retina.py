import retinasdk
import json, operator

fullClient = retinasdk.FullClient("c3412e70-f345-11e5-8378-4dad29be0fab", apiServer="http://api.cortical.io/rest", retinaName="en_associative")
#print(fullClient.getRetinas(), end="\n\n")
#print(fullClient.getTerms(term="apple",getFingerprint=True), end="\n\n")
#print(fullClient.compare(json.dumps([{"term": "synapse"}, {"term": "skylab"}])))
#print(fullClient.getSimilarTermsForTerm("android", getFingerprint=False))
#print(fullClient.getContextsForTerm("python", maxResults=3))
#print(fullClient.getImage(json.dumps({"term": "python"}), plotShape="square"))
#print(fullClient.getLanguageForText("Dieser Text ist auf Deutsch"))
#print(fullClient.getFingerprintForText("Python is a widely used general-purpose, high-level programming language."))
#print(fullClient.getFingerprintsForTexts(["first text", "apple", "google", "microsoft"]))
comparison1 = [{"term": "car"}, {"term": "car"}]
comparison2 = [{"term": "cat"}, {"text": "skylab was a space station"}]
comparison3 = [{"term": "cow"}, {"term": "buffalo"}]
#print([fullClient.compareBulk(json.dumps([comparison1, comparison2, comparison3]))[i].weightedScoring for i in range(3)])
#print(fullClient.compare(json.dumps([{"text": "How can I, as a teen, improve my life the most in a single day?"}, {"text": "How true is the storyline, settings and event description in Mahabharat TV series airing on Star Plus?"}])))
d = [] * 2
d[0].append([1,2])
d[1].append([2,3])
print(d)
