(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["vendors~ans-people~cmd-deletepqh~cmd-openpeoplecentricsb~cpl-answerproviderhost~extracted-people-ans~19412f03"],{"+DtA":function(t,e,r){"use strict";r.d(e,"a",(function(){return n}));function n(t,e){return void 0===e&&(e=!1),t?i(t.toUpperCase(),e):""}function i(t,e){var r=t.match(/^\{(.*)\}$/);return r?e?r[0]:r[1]:e?"{"+t+"}":t}},B8KU:function(t,e,r){"use strict";var n=r("x0XX"),i=r("vcbJ"),o=r("qha/"),s=r("+DtA"),a="".concat("KillSwitchOverrides","_").concat("enableKillSwitches"),h="".concat("KillSwitchOverrides","_").concat("disableKillSwitches");function c(t){try{var e=document.cookie.match("(^|;)\\s*"+t+"\\s*=\\s*([^;]+)");return e?e.pop():""}catch(t){return""}}function u(t){var e=[],r=[];for(var n in t)(t[n]?e:r).push(n);try{document.cookie="".concat(a,"=").concat(e.join(","),";path=/;samesite=none;secure;").concat(e.length?"":"expires=Thu, 01 Jan 1970 00:00:01 GMT;"),document.cookie="".concat(h,"=").concat(r.join(","),";path=/;samesite=none;secure;").concat(r.length?"":"expires=Thu, 01 Jan 1970 00:00:01 GMT;")}catch(t){}}function p(){return function(t,e,r,n){if(e)for(var i=0,o=e.split(",");i<o.length;i++){var a=o[i];t[s.a(a,!1)]=!0}if(r)for(var h=0,c=r.split(",");h<c.length;h++){a=c[h];t[s.a(a,!1)]=!1}if(n)for(var u=0,p=n.split(",");u<p.length;u++){"!"===(a=p[u])[0]?t[s.a(a.slice(1),!1)]=!1:t[s.a(a,!1)]=!0}return t}({},c(a),c(h),void 0)}var f,l={},g=function(){l=p();var t=function(){try{return location.search?location.search.substring(1):""}catch(t){return""}}(),e=Object(o.a)(t);e.enableKillSwitches,e.disableKillSwitches,e.debugKillSwitches;u(l);try{window.__debugSetKillSwitch=function(t,e){void 0===e&&(e=!0),l[s.a(t,!1)]=!!e,u(l)}}catch(t){}g=function(){}};function d(t,e){f&&f(t,e)}var _,y=function(){function t(){}return t.initKillSwitches=function(e){t._killSwitch=v({killSwitches:e||{}})},t.isActivated=function(e,r,n){var i;if(g(),t._killSwitch)return t.isActivated=t._killSwitch.isActivated,t._killSwitch.isActivated(e,r,n);try{if(window._spPageContextInfo){var o=v(window._spPageContextInfo);return(null===(i=window._spPageContextInfo)||void 0===i?void 0:i.killSwitches)&&(t.isActivated=o.isActivated),o.isActivated(e,r,n)}if(window.Flight){var s=void 0;if(window.Flight.KillSwitches){s={};for(var a=0,h=Object.keys(window.Flight.KillSwitches);a<h.length;a++){var c=h[a];s[window.Flight.KillSwitches[c]]=!0}}var u=v({killSwitches:s});return s&&(t.isActivated=u.isActivated),u.isActivated(e,r,n)}}catch(t){return d(e,!1),!1}return d(e,!1),!1},t}();function v(t){var e=t&&t.killSwitches;return{isActivated:function(t,r,n){var i=!1;if(t){var o=s.a(t,!1);o in l?i=!!l[o]:e&&(i=!!e[o])}return d(t,i),i}}}!function(t){t[t.scheme=0]="scheme",t[t.authority=1]="authority",t[t.path=2]="path",t[t.query=3]="query"}(_||(_={}));var m=/[;\/?:@&=$,]/,b=/[\/?]/;function S(t){var e={};for(var r in t)t.hasOwnProperty(r)&&(e[r.toLowerCase()]=t[r].toLowerCase());return e}var w=function(){function t(t,e){this._scheme="",this._user="",this._host="",this._port="",this._path="",this._pathSegments=[],this._pathEncoded="",this._query={},this._fragment="",e&&(this._queryCaseInsensitive=!!e.queryCaseInsensitive,this._pathCaseInsensitive=!!e.pathCaseInsensitive),this._isCatchParsePathDecodeExceptionKSActive=y.isActivated("e759149e-00c5-412d-bc6d-d63d0bb5fe84","08/24/2022","Catch decoding errors while parsing the URI path"),this._isMailtoPathSlashFixingKSActive=y.isActivated("ec2e96a7-d77c-4fb3-addd-1c430d6985b3","11/22/2022","Avoid adding slash prefix for mailto schema"),this._parseURI(t)}return t.concatenate=function(){for(var e=[],r=0;r<arguments.length;r++)e[r]=arguments[r];for(var n="",i=0;i<e.length;i++){var o=e[i];i>0&&(o=t.ensureNoPrecedingSlash(o)),i<e.length-1&&(o=t.ensureTrailingSlash(o)),n+=o}return n},t.ensureNoPrecedingSlash=function(t){return"/"===t[0]?t.substr(1):t},t.ensureTrailingSlash=function(t){return"/"!==t[t.length-1]?t+"/":t},t.prototype.getScheme=function(){return this._scheme},t.prototype.setScheme=function(t){this._scheme=t},t.prototype.getAuthority=function(){return this._getAuthority(!1)},t.prototype.setAuthority=function(t){this._parseAuthority(t)},t.prototype.getUser=function(){return this._user},t.prototype.getHost=function(){return this._host},t.prototype.getPort=function(){return this._port},t.prototype.getPath=function(t){var e=this._path;return Boolean(t)&&null!==e&&e.lastIndexOf("/")===e.length-1&&(e=e.slice(0,-1)),e},t.prototype.getLeftPart=function(t){var e=this._scheme+"://";return t===_.authority&&(e+=this.getAuthority()),t===_.path&&(e+=this.getPath()),t===_.query&&(e+=this.getQuery()),e},t.prototype.setPath=function(t){this._isMailtoPathSlashFixingKSActive?t&&"/"!==t[0]&&(t="/"+t):"mailto"!==this.getScheme().toLowerCase()&&t&&"/"!==t[0]&&(t="/"+t),this._parsePath(t)},t.prototype.getPathSegments=function(){return this._pathSegments},t.prototype.getLastPathSegment=function(){var t=this._pathSegments;return t[t.length-1]||""},t.prototype.getQuery=function(t){return this._serializeQuery(t)},t.prototype.setQuery=function(t){this.setQueryFromObject(this._deserializeQuery(t))},t.prototype.getQueryAsObject=function(){return this._query},t.prototype.setQueryFromObject=function(t){for(var e in this._query={},t)t.hasOwnProperty(e)&&this.setQueryParameter(e,t[e])},t.prototype.getQueryParameter=function(t){var e=null,r=this._query;if(this._queryCaseInsensitive)for(var n in t=t.toLowerCase(),r)r.hasOwnProperty(n)&&n.toLowerCase()===t&&(e=r[n]);else e=r[t];return e||null},t.prototype.setQueryParameter=function(t,e,r){void 0===r&&(r=!0);var n=this._decodeQueryString(e);(n||r)&&(this._query[this._decodeQueryString(t)]=n)},t.prototype.removeQueryParameter=function(t){delete this._query[this._decodeQueryString(t)]},t.prototype.getFragment=function(){return this._fragment},t.prototype.setFragment=function(t){"#"===t[0]&&(t=t.substring(1)),this._fragment=this._decodeQueryString(t)},t.prototype.equals=function(t){return Object(i.a)(this._scheme,t.getScheme())&&this._user===t.getUser()&&Object(i.a)(this._host,t.getHost())&&this._port===t.getPort()&&this._fragment===t.getFragment()&&this._equalsCaseAppropriate(this.getPath(!0),t.getPath(!0),this._pathCaseInsensitive)&&this._equalsCaseAppropriate(this.getQuery(),t.getQuery(),this._queryCaseInsensitive)},t.prototype.equivalent=function(t){return Object(i.a)(this._scheme,t.getScheme())&&Object(i.a)(this._user,t.getUser())&&Object(i.a)(this._host,t.getHost())&&Object(i.a)(this._port,t.getPort())&&Object(i.a)(this.getPath(!0),t.getPath(!0))&&Object(n.a)(S(this.getQueryAsObject()),S(t.getQueryAsObject()))&&Object(i.a)(this._fragment,t.getFragment())},t.prototype.toString=function(t){return this._getStringInternal(!0,t)},t.prototype.getDecodedStringForDisplay=function(){return this._getStringInternal(!1)},t.prototype.getStringWithoutQueryAndFragment=function(){return this._getStringWithoutQueryAndFragmentInternal(!0)},t.prototype._equalsCaseAppropriate=function(t,e,r){return r?Object(i.a)(t,e):t===e},t.prototype._getStringInternal=function(t,e){var r=this._getStringWithoutQueryAndFragmentInternal(t,e),n=this.getQuery(t);return n&&(r+="?"+n),this._fragment&&(r+="#"+(t?encodeURIComponent(this._fragment):this._fragment)),r},t.prototype._getStringWithoutQueryAndFragmentInternal=function(t,e){var r="";this._scheme&&(r+=(t?encodeURIComponent(this._scheme):this._scheme)+":");var n=this._getAuthority(t,e);return n&&(r+="//"+n),this._pathEncoded&&(r+=t?this._pathEncoded:this._path),r},t.prototype._deserializeQuery=function(t){var e={};0===t.indexOf("?")&&(t=t.substring(1));for(var r=0,n=t.split(/[;&]+/);r<n.length;r++){var i=n[r],o=i.indexOf("=");o<0&&(o=i.length),o>0&&(e[i.substr(0,o)]=i.substr(o+1))}return e},t.prototype._serializeQuery=function(t){var e="";for(var r in this._query)if(this._query.hasOwnProperty(r)){var n=r,i=this._query[r];t&&(n=encodeURIComponent(n),i=encodeURIComponent(i)),e+=null===i||""===i?n+"=&":n+"="+i+"&"}return""!==e&&(e=e.slice(0,-1)),e},t.prototype._parseURI=function(t){var e=t,r=e.indexOf("#");if(r>=0){var n=e.substring(r+1);this.setFragment(n),e=e.substring(0,r)}var i=e.search(m);if(i>=0){":"===e[i]&&(this.setScheme(e.substring(0,i)),e=e.substring(i+1));var o="";if(0===e.indexOf("//")){var s=(e=e.substring(2)).search(b);if(s>=0?(o=e.substring(0,s),e=e.substring(s)):(o=e,e=""),this.setAuthority(o),!e)return void this.setPath("")}var a=e.indexOf("?");a>=0&&(this.setQuery(e.substring(a+1)),e=e.substring(0,a)),this.setPath(e)}else this.setPath(e)},t.prototype._parseAuthority=function(t){this._host=t;var e=t.lastIndexOf("@");e>=0&&(this._host=this._host.substring(e+1));var r=this._host.indexOf(":");if(!(e<0&&r<0)){var n=t;e<0?this._host=n:(this._user=n.substring(0,e),this._host=n.substring(e+1)),r>=0&&(this._port=this._host.substring(r+1),this._host=this._host.substring(0,r)),this._user=decodeURIComponent(this._user),this._host=decodeURIComponent(this._host)}},t.prototype._parsePath=function(t){if(this._isCatchParsePathDecodeExceptionKSActive)this._path=decodeURIComponent(t);else try{this._path=decodeURIComponent(t)}catch(e){this._path=t}var e=this._pathSegments=[];this._pathEncoded=t;for(var r=t.split("/"),n=0;n<r.length;++n)if(this._isCatchParsePathDecodeExceptionKSActive)e[n]=decodeURIComponent(r[n]);else try{e[n]=decodeURIComponent(r[n])}catch(t){e[n]=r[n]}""===e[0]&&e.shift(),""===e[e.length-1]&&e.pop()},t.prototype._getAuthority=function(t,e){void 0===e&&(e={});var r,n,i,o=e&&e.doNotPercentEncodeHost,s="";return t?(r=encodeURIComponent(this._user).replace("%3A",":"),n=o?this._host:encodeURIComponent(this._host),i=encodeURIComponent(this._port)):(r=this._user,n=this._host,i=this._port),""!==r&&(s=r+"@"),""!==this._host&&(s+=n),""!==this._port&&(s+=":"+i),s},t.prototype._decodeQueryString=function(t){var e=t;try{e=decodeURIComponent(t.replace(/\+/g," "))}catch(t){}return e},t}();e.a=w},"qha/":function(t,e,r){"use strict";function n(t){var e={};if(t)for(var r=t.split("&"),n=0;n<r.length;n++){var i=r[n].split("=");void 0!==i[1]&&(i[1]=i[1].replace(/\+/g," "),e[i[0]]=decodeURIComponent(i[1]))}return e}r.d(e,"a",(function(){return n}))},vcbJ:function(t,e,r){"use strict";r.d(e,"a",(function(){return n}));function n(t,e){return t&&e?t.toUpperCase()===e.toUpperCase():t===e}},x0XX:function(t,e,r){"use strict";function n(t,e,r){var n=[],i=[],o=r||function(t,e){return t===e};return function t(e,r){if(e===r)return!0;if(null===e||null===r)return!1;if("object"!=typeof e||"object"!=typeof r)return!1;var s=Object.keys(e).sort(),a=Object.keys(r).sort();return s.length===a.length&&!!s.every((function(s,h){if(s!==a[h])return!1;if("function"==typeof e[s]||"function"==typeof r[s])return!0;if(o(e[s],r[s]))return!0;if("object"==typeof e[s]){if(-1!==n.indexOf(e[s]))throw new Error("Cannot perform DeepCompare() because a circular reference was encountered, object: ".concat(e,", ")+"property: ".concat(s));if(n.push(e[s]),-1!==i.indexOf(r[s]))throw new Error("Cannot perform DeepCompare() because a circular reference was encountered, object: ".concat(r,", ")+"property: ".concat(s));return i.push(r[s]),t(e[s],r[s])?(n.pop(),i.pop(),!0):!1}return!1}))}(t,e)}r.d(e,"a",(function(){return n}))}}]);